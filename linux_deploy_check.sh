#!/bin/bash
# Linux 部署快速检查脚本
# 使用方法: bash linux_deploy_check.sh

echo "========================================================================"
echo "🐧 Linux 部署环境检查"
echo "========================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python
echo "1️⃣  检查 Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ Python 已安装: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi

# 检查 PyTorch
echo ""
echo "2️⃣  检查 PyTorch..."
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    echo -e "${GREEN}✅ PyTorch 已安装: $TORCH_VERSION${NC}"
    echo "   CUDA 可用: $CUDA_AVAILABLE"
    
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
        echo -e "${GREEN}   🎮 CUDA 版本: $CUDA_VERSION${NC}"
        echo "   📊 GPU 数量: $GPU_COUNT"
    else
        echo -e "${YELLOW}   ⚠️  CPU 模式（训练会较慢）${NC}"
    fi
else
    echo -e "${RED}❌ PyTorch 未安装${NC}"
    echo "   安装命令: pip install torch torchvision"
    exit 1
fi

# 检查依赖
echo ""
echo "3️⃣  检查依赖包..."
DEPENDENCIES=("numpy" "scipy" "scikit-learn" "pyvista" "tqdm" "matplotlib")
MISSING_DEPS=()

for dep in "${DEPENDENCIES[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo -e "${GREEN}   ✅ $dep${NC}"
    else
        echo -e "${RED}   ❌ $dep (缺失)${NC}"
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  需要安装以下依赖:${NC}"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "   - $dep"
    done
    echo ""
    echo "安装命令:"
    echo "   pip install ${MISSING_DEPS[*]}"
    exit 1
fi

# 检查 AMP 兼容性
echo ""
echo "4️⃣  检查 AMP 兼容性..."
if python3 test_amp_compatibility.py > /dev/null 2>&1; then
    echo -e "${GREEN}✅ AMP 兼容性检查通过${NC}"
    python3 test_amp_compatibility.py 2>&1 | grep "AMP 来源" || true
else
    echo -e "${YELLOW}⚠️  AMP 测试失败（但可能仍可使用 CPU 模式）${NC}"
fi

# 检查数据集
echo ""
echo "5️⃣  检查数据集..."
if [ -d "datasets/segmentation_dataset" ]; then
    VTP_COUNT=$(find datasets/segmentation_dataset -name "*.vtp" | wc -l)
    echo -e "${GREEN}✅ 数据集目录存在${NC}"
    echo "   VTP 文件数量: $VTP_COUNT"
else
    echo -e "${RED}❌ 数据集目录不存在${NC}"
    echo "   请确保 datasets/segmentation_dataset/ 存在"
    exit 1
fi

# 检查缓存
echo ""
echo "6️⃣  检查缓存完整性..."
if [ -f "check_cache_integrity.py" ]; then
    if python3 check_cache_integrity.py > /dev/null 2>&1; then
        echo -e "${GREEN}✅ 缓存检查通过${NC}"
    else
        echo -e "${YELLOW}⚠️  发现损坏的缓存文件${NC}"
        echo "   运行以下命令修复:"
        echo "   python3 check_cache_integrity.py"
    fi
else
    echo -e "${YELLOW}⚠️  缓存检查脚本不存在${NC}"
fi

# 测试导入
echo ""
echo "7️⃣  测试训练脚本导入..."
cd src/iMeshSegNet 2>/dev/null || { echo -e "${RED}❌ src/iMeshSegNet 目录不存在${NC}"; exit 1; }
if python3 -c "import m1_train" 2>/dev/null; then
    echo -e "${GREEN}✅ m1_train.py 导入成功${NC}"
else
    echo -e "${RED}❌ m1_train.py 导入失败${NC}"
    python3 -c "import m1_train" 2>&1 | head -10
    exit 1
fi
cd ../..

# 总结
echo ""
echo "========================================================================"
echo "📋 部署检查总结"
echo "========================================================================"
echo ""

if [ ${#MISSING_DEPS[@]} -eq 0 ] && [ "$CUDA_AVAILABLE" = "True" ]; then
    echo -e "${GREEN}✅ 所有检查通过！可以开始训练${NC}"
    echo ""
    echo "🚀 启动训练命令:"
    echo "   cd src/iMeshSegNet"
    echo "   python3 m1_train.py"
    echo ""
    echo "📊 后台运行（推荐）:"
    echo "   nohup python3 m1_train.py > train.log 2>&1 &"
    echo "   tail -f train.log"
elif [ ${#MISSING_DEPS[@]} -eq 0 ] && [ "$CUDA_AVAILABLE" = "False" ]; then
    echo -e "${YELLOW}⚠️  环境正常但使用 CPU 模式${NC}"
    echo ""
    echo "CPU 训练会比较慢，建议："
    echo "  1. 使用 GPU 服务器"
    echo "  2. 减小 batch_size"
    echo "  3. 减小 sample_cells"
    echo ""
    echo "如果接受慢速训练，可以运行:"
    echo "   cd src/iMeshSegNet"
    echo "   python3 m1_train.py"
else
    echo -e "${RED}❌ 存在问题，请修复后重试${NC}"
    exit 1
fi

echo ""
echo "========================================================================"
