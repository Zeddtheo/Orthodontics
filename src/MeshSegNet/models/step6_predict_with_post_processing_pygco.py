import os
import numpy as np
import torch
import torch.nn as nn
from meshsegnet import *
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
import scipy.io as sio
import shutil
import time
from sklearn.svm import SVC # uncomment this line if you don't install thudersvm
# from thundersvm import SVC 
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import warnings
try:
    from pygco import cut_from_graph
except Exception as e:
    warnings.warn(f'pygco unavailable ({e}); falling back to unary argmin without graph cut smoothing.')

    def cut_from_graph(edges, unaries, pairwise):
        return np.argmin(unaries, axis=1).astype(np.int32)
import utils

LABEL_COLORS = {
    0:  [160, 160, 160],
    1:  [255,  69,   0],
    2:  [255, 165,   0],
    3:  [255, 215,   0],
    4:  [154, 205,  50],
    5:  [ 34, 139,  34],
    6:  [ 46, 139,  87],
    7:  [ 72, 209, 204],
    8:  [ 70, 130, 180],
    9:  [ 65, 105, 225],
    10: [138,  43, 226],
    11: [199,  21, 133],
    12: [255, 105, 180],
    13: [205,  92,  92],
    14: [255, 140,   0],
    15: [255, 228, 196],
}
_DEFAULT_COLOR = np.array([90, 90, 90], dtype=np.uint8)

MODEL_FILES = {
    'U': 'MeshSegNet_Max_15_classes_72samples_lr1e-2_best.pth',
    'L': 'MeshSegNet_Man_15_classes_72samples_lr1e-2_best.pth',
}

_BASE_MIRROR_PAIRS = (
    (1, 14),
    (2, 13),
    (3, 12),
    (4, 11),
    (5, 10),
    (6, 9),
    (7, 8),
)
MIRROR_LABEL_MAP = {
    arch: {
        0: 0,
        **{a: b for a, b in _BASE_MIRROR_PAIRS},
        **{b: a for a, b in _BASE_MIRROR_PAIRS},
    }
    for arch in ('U', 'L')
}


def _mirror_permutation(arch: str, num_classes: int) -> np.ndarray:
    perm = np.arange(num_classes, dtype=np.int32)
    mapping = MIRROR_LABEL_MAP.get(arch.upper())
    if not mapping:
        return perm
    for src, dst in mapping.items():
        if 0 <= src < num_classes and 0 <= dst < num_classes:
            perm[src] = dst
    return perm


def _safe_cut_from_graph(edges: np.ndarray, unaries: np.ndarray, pairwise: np.ndarray, stage: str) -> np.ndarray:
    if unaries.ndim != 2:
        raise ValueError(f'[{stage}] Unaries must be 2-D, got shape {unaries.shape}')
    if pairwise.ndim != 2 or pairwise.shape[0] != pairwise.shape[1]:
        raise ValueError(f'[{stage}] Pairwise must be square, got shape {pairwise.shape}')
    if unaries.shape[1] != pairwise.shape[0]:
        raise ValueError(f'[{stage}] Incompatible unary/pairwise shapes: {unaries.shape} vs {pairwise.shape}')
    if edges.ndim != 2 or edges.shape[1] != 3:
        raise ValueError(f'[{stage}] Edges must be of shape (E, 3), got {edges.shape}')

    if unaries.size == 0:
        return np.empty(0, dtype=np.int32)
    if edges.size == 0:
        return np.argmin(unaries, axis=1).astype(np.int32)

    edges_c = np.ascontiguousarray(edges, dtype=np.int32)
    unaries_c = np.ascontiguousarray(unaries, dtype=np.int32)
    pairwise_c = np.ascontiguousarray(pairwise, dtype=np.int32)
    try:
        labels = cut_from_graph(edges_c, unaries_c, pairwise_c)
        labels = np.asarray(labels, dtype=np.int32)
        if labels.shape[0] != unaries.shape[0]:
            raise ValueError(f'[{stage}] pygco returned {labels.shape[0]} labels, expected {unaries.shape[0]}')
        return labels
    except Exception as exc:
        warnings.warn(f'[{stage}] pygco failed ({exc}); falling back to unary argmin.')
        return np.argmin(unaries, axis=1).astype(np.int32)


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32, copy=False).reshape(-1)
    rgb = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for lab in np.unique(labels):
        rgb[labels == lab] = np.asarray(LABEL_COLORS.get(int(lab), _DEFAULT_COLOR), dtype=np.uint8)
    return rgb


def attach_label_data(mesh_obj, labels: np.ndarray):
    labels = labels.astype(np.int32, copy=False).reshape(-1)
    colors = labels_to_rgb(labels)
    mesh_obj.celldata['Label'] = labels
    mesh_obj.celldata['PredictedID'] = labels
    mesh_obj.celldata['RGB'] = colors

    n_points = mesh_obj.npoints
    point_labels = np.zeros(n_points, dtype=np.int32)
    point_colors = np.zeros((n_points, 3), dtype=np.uint8)

    faces = np.asarray(mesh_obj.cells, dtype=np.int32)
    adjacency = [[] for _ in range(n_points)]
    for cid, face in enumerate(faces):
        for pid in face:
            adjacency[pid].append(cid)

    for pid, cid_list in enumerate(adjacency):
        if not cid_list:
            continue
        labs = labels[cid_list]
        uniq, cnt = np.unique(labs, return_counts=True)
        lab = int(uniq[np.argmax(cnt)])
        point_labels[pid] = lab
        point_colors[pid] = np.asarray(LABEL_COLORS.get(lab, _DEFAULT_COLOR), dtype=np.uint8)

    mesh_obj.pointdata['Label'] = point_labels
    mesh_obj.pointdata['PredictedID'] = point_labels
    mesh_obj.pointdata['RGB'] = point_colors
    return mesh_obj


def infer_arch_from_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    tokens = base.replace('-', '_').split('_')
    for token in reversed(tokens):
        if token in {'u', 'upper', 'max', 'maxilla'}:
            return 'U'
        if token in {'l', 'lower', 'man', 'mandible'}:
            return 'L'
    if base.endswith('u'):
        return 'U'
    if base.endswith('l'):
        return 'L'
    raise ValueError(f"无法从文件名 {filename} 推断颌别（需包含 U/L 或 upper/lower 标记）")


if __name__ == '__main__':

    if torch.cuda.is_available():
        gpu_id = utils.get_avail_gpu()
        torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)
    else:
        gpu_id = None

    upsampling_method = os.environ.get('MESHSEGNET_UPSAMPLING', 'KNN').upper()
    if upsampling_method not in {'SVM', 'KNN'}:
        upsampling_method = 'KNN'

    model_path = os.environ.get('MESHSEGNET_MODEL_PATH', './src/MeshSegNet/models')

    mesh_path = os.environ.get('MESHSEGNET_MESH_PATH', './API/tests')
    samples_env = os.environ.get('MESHSEGNET_SAMPLES')
    if samples_env:
        sample_filenames = [s.strip() for s in samples_env.split(',') if s.strip()]
    else:
        sample_filenames = sorted([
            fname for fname in os.listdir(mesh_path)
            if fname.lower().endswith('.stl')
        ])
    output_path = os.environ.get('MESHSEGNET_OUTPUT_PATH', './outputs')
    os.makedirs(output_path, exist_ok=True)

    num_classes = 15
    num_channels = 15

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cache = {}

    def get_model_for_arch(arch: str) -> nn.Module:
        arch = arch.upper()
        if arch not in MODEL_FILES:
            raise ValueError(f'未知颌别 {arch}，无法选择模型')
        if arch not in model_cache:
            ckpt_path = os.path.join(model_path, MODEL_FILES[arch])
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f'缺少模型权重 {ckpt_path}')
            model_instance = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model_instance.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
            model_instance = model_instance.to(device, dtype=torch.float)
            model_instance.eval()
            model_cache[arch] = model_instance
        return model_cache[arch]

    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


    # Predicting
    with torch.no_grad():
        for i_sample in sample_filenames:

            arch = infer_arch_from_name(i_sample)
            model = get_model_for_arch(arch)

            start_time = time.time()
            # create tmp folder
            tmp_path = './.tmp/'
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            print('Predicting Sample filename: {}'.format(i_sample))
            # read image and label (annotation)
            mesh = vedo.load(os.path.join(mesh_path, i_sample))
            mesh_original = mesh.clone()

            # pre-processing: downsampling
            print('\tDownsampling...')
            target_num = 20000
            ratio = target_num/mesh.ncells # calculate ratio
            mesh_d = mesh.clone()
            mesh_d.decimate(fraction=ratio)
            predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)

            # move mesh to origin
            print('\tPredicting...')
            points = mesh_d.points.copy()
            mean_cell_centers = mesh_d.center_of_mass()
            points[:, 0:3] -= mean_cell_centers[0:3]

            ids = np.array(mesh_d.cells, dtype=np.int32)
            cells = points[ids].reshape(mesh_d.ncells, 9).astype(dtype='float32')

            # customized normal calculation; the vtk/vedo build-in function will change number of points
            mesh_d.compute_normals()
            normals_array = np.asarray(mesh_d.celldata['Normals'], dtype=np.float32)

            # move mesh to origin
            barycenters_centered = mesh_d.cell_centers().points.copy()
            barycenters_centered -= mean_cell_centers[0:3]

            base_points = np.asarray(points, dtype=np.float32)
            base_normals = normals_array.copy()

            def _unit_normals(arr: np.ndarray) -> np.ndarray:
                arr_copy = arr.copy()
                norms = np.linalg.norm(arr_copy, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr_copy /= norms
                return arr_copy

            base_normals = _unit_normals(base_normals)

            def _coarse_forward(points_input: np.ndarray, normals_input: np.ndarray) -> np.ndarray:
                pts = np.array(points_input, dtype=np.float32, copy=True)
                norms = np.array(normals_input, dtype=np.float32, copy=True)
                cell_vertices = pts[ids].reshape(mesh_d.ncells, 3, 3)
                cells_flat = cell_vertices.reshape(mesh_d.ncells, 9)
                bary_local = cell_vertices.mean(axis=1)

                maxs = pts.max(axis=0)
                mins = pts.min(axis=0)
                means = pts.mean(axis=0)
                stds = pts.std(axis=0)
                nmeans = norms.mean(axis=0)
                nstds = norms.std(axis=0)

                stds[stds == 0] = 1.0
                spans = maxs - mins
                spans[spans == 0] = 1.0
                nstds[nstds == 0] = 1.0

                for i in range(3):
                    cells_flat[:, i] = (cells_flat[:, i] - means[i]) / stds[i] # point 1
                    cells_flat[:, i+3] = (cells_flat[:, i+3] - means[i]) / stds[i] # point 2
                    cells_flat[:, i+6] = (cells_flat[:, i+6] - means[i]) / stds[i] # point 3
                    bary_local[:, i] = (bary_local[:, i] - mins[i]) / spans[i]
                    norms[:, i] = (norms[:, i] - nmeans[i]) / nstds[i]

                X_local = np.column_stack((cells_flat, bary_local, norms))

                # computing A_S and A_L
                A_S_local = np.zeros([X_local.shape[0], X_local.shape[0]], dtype='float32')
                A_L_local = np.zeros([X_local.shape[0], X_local.shape[0]], dtype='float32')
                D = distance_matrix(X_local[:, 9:12], X_local[:, 9:12])
                A_S_local[D<0.1] = 1.0
                A_S_local = A_S_local / np.dot(np.sum(A_S_local, axis=1, keepdims=True), np.ones((1, X_local.shape[0])))

                A_L_local[D<0.2] = 1.0
                A_L_local = A_L_local / np.dot(np.sum(A_L_local, axis=1, keepdims=True), np.ones((1, X_local.shape[0])))

                # numpy -> torch.tensor
                X_tensor = torch.from_numpy(X_local.transpose(1, 0)[None, :, :]).to(device, dtype=torch.float)
                A_S_tensor = torch.from_numpy(A_S_local.reshape(1, A_S_local.shape[0], A_S_local.shape[1])).to(device, dtype=torch.float)
                A_L_tensor = torch.from_numpy(A_L_local.reshape(1, A_L_local.shape[0], A_L_local.shape[1])).to(device, dtype=torch.float)

                tensor_prob = model(X_tensor, A_S_tensor, A_L_tensor).to(device, dtype=torch.float)
                return tensor_prob.cpu().numpy()

            patch_prob_output = _coarse_forward(base_points, base_normals)

            def _apply_transform(matrix: np.ndarray):
                pts_t = base_points @ matrix.T
                norms_t = _unit_normals(base_normals @ matrix.T)
                return pts_t, norms_t

            MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
            ROT_Z_180 = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)

            mirror_probs = _coarse_forward(*_apply_transform(MIRROR_X))
            rot_probs = _coarse_forward(*_apply_transform(ROT_Z_180))

            def _build_edges(cell_ids: np.ndarray, normals: np.ndarray, barycenters: np.ndarray, lambda_c_val: float, scale_val: int) -> np.ndarray:
                n_cells = cell_ids.shape[0]
                edge_records = []
                for i_node in range(n_cells):
                    nei = np.sum(np.isin(cell_ids, cell_ids[i_node]), axis=1)
                    nei_idx = np.where(nei == 2)[0]
                    for j in nei_idx:
                        if i_node >= j:
                            continue
                        ni = normals[i_node]
                        nj = normals[j]
                        denom = np.linalg.norm(ni) * np.linalg.norm(nj)
                        if denom <= 0.0:
                            cos_theta = 0.9999
                        else:
                            cos_theta = float(np.dot(ni, nj) / denom)
                            cos_theta = max(min(cos_theta, 0.9999), -0.9999)
                        theta = np.arccos(cos_theta)
                        ratio = max(theta / np.pi, 1.0e-6)
                        phi = np.linalg.norm(barycenters[i_node] - barycenters[j])
                        if theta > np.pi / 2.0:
                            weight = -np.log10(ratio) * phi
                        else:
                            beta = 1 + abs(np.dot(ni, nj))
                            weight = -beta * np.log10(ratio) * phi
                        weight = max(weight, 1.0)
                        edge_records.append((i_node, j, weight))
                if not edge_records:
                    return np.zeros((0, 3), dtype=np.int32)
                edges_arr = np.asarray(edge_records, dtype=np.float32)
                edges_arr[:, 2] *= lambda_c_val * scale_val
                return edges_arr.astype(np.int32)

            def _neg_log_top1(probs: np.ndarray) -> float:
                prob_matrix = np.squeeze(probs, axis=0)
                top1_idx = np.argmax(prob_matrix, axis=-1)
                top1_prob = prob_matrix[np.arange(top1_idx.shape[0]), top1_idx]
                return float(-np.log(np.clip(top1_prob, 1.0e-12, 1.0)).sum())

            orientation_scores = {
                'original': _neg_log_top1(patch_prob_output),
                'mirror_x': _neg_log_top1(mirror_probs),
                'rot_z_180': _neg_log_top1(rot_probs),
            }
            best_orientation = min(orientation_scores, key=orientation_scores.get)
            print('\tOrientation NLL (lower is better): {}'.format(
                ', '.join(f'{k}={v:.4f}' for k, v in orientation_scores.items())))
            print(f'\tOrientation closest to training set: {best_orientation}')

            orientation_prob_map = {
                'original': patch_prob_output,
                'mirror_x': mirror_probs,
                'rot_z_180': rot_probs,
            }
            chosen_probs = np.squeeze(orientation_prob_map[best_orientation], axis=0).copy()
            if best_orientation in {'mirror_x', 'rot_z_180'}:
                perm = _mirror_permutation(arch, chosen_probs.shape[1])
                chosen_probs = chosen_probs[:, perm]
            coarse_probs = np.clip(chosen_probs, 1.0e-6, 1.0)
            patch_prob_output = coarse_probs[None, ...]
            predicted_labels_d = np.argmax(coarse_probs, axis=-1).reshape(-1, 1)

            # output downsampled predicted labels
            mesh2 = mesh_d.clone()
            mesh2 = attach_label_data(mesh2, predicted_labels_d)
            vedo.write(mesh2, os.path.join(output_path, '{}_d_predicted.vtp'.format(i_sample[:-4])))

            # refinement
            print('\tRefining by pygco...')
            scale = 70
            coarse_unaries = (-scale * np.log(coarse_probs)).astype(np.int32)

            compat = np.ones((num_classes, num_classes), dtype=np.int32)
            np.fill_diagonal(compat, 0)
            pairwise = np.ascontiguousarray(compat, dtype=np.int32)

            background_bias = int(scale * 0.4)
            coarse_unaries[:, 0] += background_bias
            coarse_unaries = np.ascontiguousarray(coarse_unaries, dtype=np.int32)

            normals_coarse_graph = np.asarray(mesh_d.celldata['Normals'], dtype=np.float32).copy()
            barycenters_coarse_graph = mesh_d.cell_centers().points.copy()
            cell_ids_coarse = np.asarray(mesh_d.cells, dtype=np.int32)

            lambda_c = 12
            print(f'\t[params][coarse] scale={scale}, lambda_c={lambda_c}')
            print(f'\t[GC][coarse] use pygco = {cut_from_graph.__module__ != "__main__"}')
            coarse_edges = _build_edges(cell_ids_coarse, normals_coarse_graph, barycenters_coarse_graph, lambda_c, scale)
            coarse_edges = np.ascontiguousarray(coarse_edges, dtype=np.int32)
            if coarse_edges.size:
                print('\t[stats] coarse_unaries range [{}, {}]'.format(coarse_unaries.min(), coarse_unaries.max()))
                print('\t[stats] coarse_edges count {}, weight range [{}, {}]'.format(coarse_edges.shape[0], coarse_edges[:, 2].min(), coarse_edges[:, 2].max()))
            else:
                print('\t[stats] coarse_edges empty')

            if coarse_unaries.size:
                coarse_labels = _safe_cut_from_graph(coarse_edges, coarse_unaries, pairwise, 'coarse')
            else:
                coarse_labels = np.zeros(mesh_d.ncells, dtype=np.int32)
            refine_labels = coarse_labels.reshape([-1, 1])

            barycenters_for_stats = barycenters_centered.copy()
            label_flat = refine_labels.reshape(-1)
            if barycenters_for_stats.shape[0] == label_flat.shape[0]:
                mean_x_by_label = {}
                unique_labels = np.unique(label_flat)
                for lab in unique_labels:
                    mask = label_flat == lab
                    if mask.any():
                        mean_x_by_label[int(lab)] = float(barycenters_for_stats[mask, 0].mean())
                if mean_x_by_label:
                    print('\t[MirrorCheck] Mean centered X per label (ascending):')
                    for lab, mean_x in sorted(mean_x_by_label.items(), key=lambda kv: kv[1]):
                        print(f'\t    label {lab:2d}: mean_x={mean_x:.4f}')
                else:
                    print('\t[MirrorCheck] No labels available for barycenter statistics.')
            else:
                print('\t[MirrorCheck] Skipped barycenter stats due to shape mismatch.')

            # output refined result
            mesh3 = mesh_d.clone()
            mesh3 = attach_label_data(mesh3, refine_labels)
            vedo.write(mesh3, os.path.join(output_path, '{}_d_predicted_refined.vtp'.format(i_sample[:-4])))

            # upsampling
            print('\tUpsampling...')
            if mesh.ncells > 45000:
                target_num = 32000 # set max number of cells for fine GC stability
                ratio = target_num/mesh.ncells # calculate ratio
                mesh.decimate(fraction=ratio)
                print('Original contains too many cells, simpify to {} cells'.format(mesh.ncells))

            mesh.compute_normals()
            fine_barycenters = mesh.cell_centers().points.copy()
            normals_fine_graph = np.asarray(mesh.celldata['Normals'], dtype=np.float32).copy()

            coarse_barycenters = mesh3.cell_centers().points.copy()
            normals_coarse_feats = _unit_normals(np.asarray(mesh3.celldata['Normals'], dtype=np.float32))
            normals_fine_feats = _unit_normals(normals_fine_graph)

            bary_min = coarse_barycenters.min(axis=0)
            bary_span = coarse_barycenters.max(axis=0) - bary_min
            bary_span[bary_span == 0] = 1.0

            coarse_centers_norm = (coarse_barycenters - bary_min) / bary_span
            fine_centers_norm = (fine_barycenters - bary_min) / bary_span

            gum_label = 0
            alpha = 3.0
            coarse_labels_flat = refine_labels.reshape(-1)
            coarse_is_gum = (coarse_labels_flat == gum_label).astype(np.float32)
            coarse_features = np.hstack([
                coarse_centers_norm,
                alpha * normals_coarse_feats,
                coarse_is_gum[:, None],
            ])

            nn_lookup = NearestNeighbors(n_neighbors=1)
            nn_lookup.fit(coarse_centers_norm)
            nearest_idx = nn_lookup.kneighbors(fine_centers_norm, return_distance=False).reshape(-1)
            fine_is_gum_hint = coarse_is_gum[nearest_idx][:, None]
            fine_features = np.hstack([
                fine_centers_norm,
                alpha * normals_fine_feats,
                fine_is_gum_hint,
            ])

            knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
            knn.fit(coarse_features, coarse_probs)
            fine_probs = np.clip(knn.predict(fine_features), 1.0e-10, 1.0)
            fine_probs = fine_probs / np.maximum(fine_probs.sum(axis=1, keepdims=True), 1.0e-12)
            fine_argmax_labels = np.argmax(fine_probs, axis=1).astype(np.int32)

            fine_unaries = (-scale * np.log(fine_probs)).astype(np.int32)
            gum_bias = int(0.6 * scale)
            fine_unaries[:, 0] += gum_bias
            fine_unaries = np.ascontiguousarray(fine_unaries, dtype=np.int32)

            print(f'\t[params][fine] scale={scale}, lambda_c={lambda_c}')
            print(f'\t[GC][fine] use pygco = {cut_from_graph.__module__ != "__main__"}')
            cell_ids_fine = np.asarray(mesh.cells, dtype=np.int32)
            print(f'\t[stats] fine mesh cells shape {cell_ids_fine.shape}, probs shape {fine_probs.shape}')
            fine_edges = _build_edges(cell_ids_fine, normals_fine_graph, fine_barycenters, lambda_c, scale)
            fine_edges = np.ascontiguousarray(fine_edges, dtype=np.int32)
            fine_labels_array: np.ndarray
            if fine_unaries.size and fine_edges.size:
                deg = np.zeros(cell_ids_fine.shape[0], dtype=np.int32)
                for edge_u, edge_v, _ in fine_edges:
                    deg[edge_u] += 1
                    deg[edge_v] += 1
                isolated = np.where(deg == 0)[0]
                print('\t[stats] fine_unaries range [{}, {}]'.format(fine_unaries.min(), fine_unaries.max()))
                print('\t[stats] fine_edges count {}, weight range [{}, {}]'.format(fine_edges.shape[0], fine_edges[:, 2].min(), fine_edges[:, 2].max()))
                print('\t[stats] fine degree stats min={}, max={}, isolated={}'.format(deg.min(), deg.max(), isolated.size))

                valid_idx = np.where(deg > 0)[0]
                if valid_idx.size:
                    reindex = -np.ones(cell_ids_fine.shape[0], dtype=np.int32)
                    reindex[valid_idx] = np.arange(valid_idx.size, dtype=np.int32)
                    reduced_edges = fine_edges.copy()
                    reduced_edges[:, 0] = reindex[reduced_edges[:, 0]]
                    reduced_edges[:, 1] = reindex[reduced_edges[:, 1]]
                    mask_valid_edges = (reduced_edges[:, 0] >= 0) & (reduced_edges[:, 1] >= 0)
                    reduced_edges = reduced_edges[mask_valid_edges]
                    print('\t[stats] fine valid nodes {}, reduced edges {}'.format(valid_idx.size, reduced_edges.shape[0]))
                    reduced_edges = np.ascontiguousarray(reduced_edges, dtype=np.int32)
                    reduced_unaries = np.ascontiguousarray(fine_unaries[valid_idx], dtype=np.int32)
                    gc_labels = _safe_cut_from_graph(reduced_edges, reduced_unaries, pairwise, 'fine')
                    fine_labels_array = np.empty(cell_ids_fine.shape[0], dtype=np.int32)
                    fine_labels_array[valid_idx] = gc_labels
                else:
                    fine_labels_array = np.empty(cell_ids_fine.shape[0], dtype=np.int32)

                if isolated.size:
                    iso_labels = np.argmin(fine_unaries[isolated], axis=1).astype(np.int32)
                    fine_labels_array[isolated] = iso_labels
            elif fine_unaries.size:
                print('\t[stats] fine_unaries range [{}, {}], no edges'.format(fine_unaries.min(), fine_unaries.max()))
                fine_labels_array = np.argmin(fine_unaries, axis=1).astype(np.int32)
            else:
                print('\t[stats] fine_unaries empty')
                fine_labels_array = np.zeros(cell_ids_fine.shape[0], dtype=np.int32)

            fine_labels = fine_labels_array.reshape(-1)
            gc_only_labels_fine = fine_labels.copy()

            # ---- fill small gum holes on fine mesh ----
            fine_faces = np.asarray(mesh.cells, dtype=np.int32)
            lab = fine_labels.copy()
            gum_label = 0

            adjacency = [[] for _ in range(mesh.ncells)]
            edge_owner = {}
            for ci, face in enumerate(fine_faces):
                for e in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    e = tuple(sorted(e))
                    if e in edge_owner:
                        cj = edge_owner[e]
                        adjacency[ci].append(cj)
                        adjacency[cj].append(ci)
                    else:
                        edge_owner[e] = ci

            # narrow band dilation guided by probabilities
            has_fine_probs = fine_probs is not None and fine_probs.shape[0] == lab.shape[0]
            if has_fine_probs:
                new_lab = lab.copy()
                log_threshold = 0.6
                for idx, current in enumerate(lab):
                    if current != gum_label:
                        continue
                    unique_neighbors = {lab[v] for v in adjacency[idx] if lab[v] != gum_label}
                    if len(unique_neighbors) != 1:
                        continue
                    candidate_label = next(iter(unique_neighbors))
                    probs_row = fine_probs[idx]
                    gum_p = max(probs_row[gum_label], 1.0e-12)
                    tooth_slice = probs_row[1:]
                    if tooth_slice.size == 0:
                        continue
                    tooth_idx = int(np.argmax(tooth_slice) + 1)
                    if tooth_idx != candidate_label:
                        continue
                    tooth_p = max(probs_row[tooth_idx], 1.0e-12)
                    if np.log(tooth_p) - np.log(gum_p) > log_threshold:
                        new_lab[idx] = candidate_label
                lab = new_lab

            visited_gum = np.zeros(mesh.ncells, dtype=bool)
            max_hole = 50
            for start in range(mesh.ncells):
                if visited_gum[start] or lab[start] != gum_label:
                    continue
                comp = [start]
                visited_gum[start] = True
                queue = [start]
                while queue:
                    u = queue.pop()
                    for v in adjacency[u]:
                        if not visited_gum[v] and lab[v] == gum_label:
                            visited_gum[v] = True
                            queue.append(v)
                            comp.append(v)
                if len(comp) > max_hole:
                    continue
                neighbor_labels = {int(lab[v]) for u in comp for v in adjacency[u] if lab[v] != gum_label}
                if len(neighbor_labels) == 1:
                    maj = neighbor_labels.pop()
                    lab[np.asarray(comp, dtype=np.int32)] = maj

            if num_classes > 1:
                visited_tooth = np.zeros(mesh.ncells, dtype=bool)
                components = {}
                for start in range(mesh.ncells):
                    if visited_tooth[start]:
                        continue
                    label_here = lab[start]
                    visited_tooth[start] = True
                    queue = [start]
                    comp = [start]
                    neighbor_labels = []
                    while queue:
                        u = queue.pop()
                        for v in adjacency[u]:
                            if lab[v] == label_here:
                                if not visited_tooth[v]:
                                    visited_tooth[v] = True
                                    queue.append(v)
                                    comp.append(v)
                            else:
                                neighbor_labels.append(lab[v])
                    components.setdefault(label_here, []).append((comp, neighbor_labels))

                for label_val, entries in components.items():
                    if label_val == gum_label or len(entries) <= 1:
                        continue
                    sizes = [len(comp) for comp, _ in entries]
                    keep_idx = int(np.argmax(sizes))
                    total_cells = float(sum(sizes))
                    for idx_entry, (comp, neighbors) in enumerate(entries):
                        if idx_entry == keep_idx:
                            continue
                        comp_indices = np.asarray(comp, dtype=np.int32)
                        comp_size = comp_indices.size
                        keep_by_size = (comp_size >= 0.25 * total_cells) or (comp_size >= 150)
                        keep_by_prob = False
                        if not keep_by_size and has_fine_probs and comp_size:
                            tooth_probs = fine_probs[comp_indices, label_val]
                            gum_probs = fine_probs[comp_indices, gum_label]
                            tooth_mean = float(np.clip(tooth_probs.mean(), 1.0e-12, 1.0))
                            gum_mean = float(np.clip(gum_probs.mean(), 1.0e-12, 1.0))
                            keep_by_prob = (np.log(tooth_mean) - np.log(gum_mean)) > 0.4
                        if keep_by_size or keep_by_prob:
                            continue
                        neighbor_labels = [n for n in neighbors if n != label_val]
                        if neighbor_labels:
                            replacement = int(np.bincount(np.asarray(neighbor_labels, dtype=np.int32)).argmax())
                        else:
                            replacement = label_val
                        lab[comp_indices] = replacement

            labels_flat = lab
            if fine_barycenters.shape[0] == labels_flat.shape[0]:
                tooth_labels = [lbl for lbl in np.unique(labels_flat) if lbl > gum_label]
                if tooth_labels:
                    max_tooth_id = num_classes - 1
                    tooth_means = {
                        lbl: float(fine_barycenters[labels_flat == lbl, 0].mean())
                        for lbl in tooth_labels
                    }
                    sorted_labels = sorted(tooth_labels, key=lambda lbl: tooth_means[lbl])
                    mapping = {gum_label: gum_label}
                    for new_idx, lbl in enumerate(sorted_labels, start=1):
                        mapping[lbl] = min(new_idx, max_tooth_id)
                    remapped = labels_flat.copy()
                    for old_lbl, new_lbl in mapping.items():
                        remapped[labels_flat == old_lbl] = new_lbl
                    labels_flat = remapped
            max_tooth_id = num_classes - 1
            for _ in range(5):
                changed = False
                label_counts = np.bincount(labels_flat, minlength=num_classes)
                for cell_idx, neighbors in enumerate(adjacency):
                    current_label = labels_flat[cell_idx]
                    if current_label <= gum_label:
                        continue
                    for neigh_idx in neighbors:
                        neigh_label = labels_flat[neigh_idx]
                        if neigh_label <= gum_label or neigh_label == current_label:
                            continue
                        if abs(current_label - neigh_label) <= 1:
                            continue
                        if label_counts[current_label] <= label_counts[neigh_label]:
                            src_label = current_label
                            target_label = neigh_label - 1 if neigh_label > current_label else neigh_label + 1
                        else:
                            src_label = neigh_label
                            target_label = current_label - 1 if current_label > neigh_label else current_label + 1
                        target_label = max(1, min(max_tooth_id, target_label))
                        if target_label == src_label:
                            continue
                        labels_flat[labels_flat == src_label] = target_label
                        label_counts = np.bincount(labels_flat, minlength=num_classes)
                        changed = True
                if not changed:
                    break

            refined_labels_fine = labels_flat.astype(np.int32)
            fine_mesh_output = attach_label_data(mesh.clone(), refined_labels_fine)
            vedo.write(fine_mesh_output, os.path.join(output_path, '{}_fine_predicted_refined.vtp'.format(i_sample[:-4])))

            projector = NearestNeighbors(n_neighbors=1)
            projector.fit(fine_barycenters)
            orig_barycenters = mesh_original.cell_centers().points.copy()
            nn_idx = projector.kneighbors(orig_barycenters, return_distance=False).reshape(-1)

            argmax_original = fine_argmax_labels[nn_idx]
            gc_only_original = gc_only_labels_fine[nn_idx]
            refined_original = refined_labels_fine[nn_idx]

            mesh_argmax = attach_label_data(mesh_original.clone(), argmax_original)
            vedo.write(mesh_argmax, os.path.join(output_path, '{}_fine_argmax.vtp'.format(i_sample[:-4])))

            mesh_gc_only = attach_label_data(mesh_original.clone(), gc_only_original)
            vedo.write(mesh_gc_only, os.path.join(output_path, '{}_gc_only.vtp'.format(i_sample[:-4])))

            mesh_refined = attach_label_data(mesh_original.clone(), refined_original)
            vedo.write(mesh_refined, os.path.join(output_path, '{}_predicted_refined.vtp'.format(i_sample[:-4])))

            #remove tmp folder
            shutil.rmtree(tmp_path)

            end_time = time.time()
            print('Sample filename: {} completed'.format(i_sample))
            print('\tcomputing time: {0:.2f} sec'.format(end_time-start_time))
