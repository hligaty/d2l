package org.example;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

/**
 * Unit test for simple App.
 */
public class NDArrayTest extends BaseTest {


    @Test
    void test() {
        //// 半开区间，从 0 ~ 11 的一维张量
        //NDArray ndArray = manager.arange(12);
        //println("shape [12]", ndArray);
        //// 元素总数为 12
        //println("ndArray numel: " + ndArray.size());
        //// 改变张量的形状而不改变元素数量和元素值
        //ndArray = ndArray.reshape(3, 4);
        //println("reshape shape(3, 4)", ndArray);
        //// 全零的 2 个 3 行 4 列 的二维数组组成的三维数组，以及全 1 的
        //println("zeros 2,3,4", manager.zeros(new Shape(2, 3, 4)));
        //println("zeros 2,3,4", manager.ones(new Shape(2, 3, 4)));
        //// 为张量的每个元素赋值赋值来确定值
        //println("all customize", manager.create(new int[][]{
        //        new int[]{2, 1, 4, 3},
        //        new int[]{1, 2, 3, 4},
        //        new int[]{4, 3, 2, 1}
        //}));
        //// 加减乘除幂指
        //NDArray x = manager.create(new float[]{1.f, 2.f, 4.f, 8.f});
        //NDArray y = manager.create(new float[]{2.f, 2.f, 2.f, 2.f});
        //println("x + y", x.add(y));
        //println("x - y", x.sub(y));
        //println("x * y", x.mul(y));
        //println("x / y", x.div(y));
        //println("x ** y", x.pow(y));
        //println("x exp", x.exp());
        //// 多个张量连结在一起，0 维合并（行合并），1 维合并（列合并）
        //x = manager.arange(0, 12, 1, DataType.FLOAT32).reshape(3, 4);
        //y = manager.create(new float[][]{
        //        new float[]{2.0f, 1, 4, 3},
        //        new float[]{1, 2, 3, 4},
        //        new float[]{4, 3, 2, 1}
        //});
        //println("row concat", x.concat(y, 0));
        //println("column concat", x.concat(y, 1));
        //// 逻辑运算符构建二元张量
        //println("eq", x.eq(y));
        //// 对张量所有元素求和产生一个只有一个元素的张量
        //println("x sum", x.sum());
        //// numpy 广播机制，只有在维度相同（下面是 2 维）且都有一个为 1
        //NDArray a = manager.arange(3).reshape(3, 1);
        //NDArray b = manager.arange(2).reshape(1, 2);
        //println("a", a);
        //println("b", b);
        //println("a + b, broadcasting mechanism", a.add(b));
        //// 通过指定索引来将元素写入矩阵（索引为 -1 时表示最后一个）
        //x.set(new NDIndex(1, 2), 9);
        //println("write value by index", x);
        //// 为多个元素赋值相同的值，我们只需要索引所有元素，然后为它们赋值
        //x.set(new NDIndex("0:2, :"), 12);
        //println("write any value by index", x);
        //// 运行一些操作可能会导致为新结果分配内存
        //NDArray before = y;
        //NDArray Y = y.add(x);
        //println("new == before ? " + (Y == before));
        //// 不重新分配内存，执行原地操作。如果在后续计算中没有重复使用 X，就可以这样来减少内存开销
        //NDArray Z = y.zerosLike();
        //println("Z", Z);
        //NDArray iZ = Z.addi(x).addi(y);
        //println("new == before ? " + (Z == iZ));
        //// 矩阵转置
        //NDArray A = manager.arange(20).reshape(5, 4);
        //NDArray AT = A.transpose();
        //println("AT", AT);
        //// 对称矩阵 A 等于其转置 A = AT
        //NDArray B = manager.create(new int[][]{
        //        new int[]{1, 2, 3},
        //        new int[]{2, 0, 4},
        //        new int[]{3, 4, 5}
        //});
        //println("B", B);
        //NDArray BT = B.transpose();
        //println("B = BT ?" + B.eq(BT));
        //// 给定具有相同形状的任何两个张量，任何按元素二元运算的结果都将是相同形状的张量
        //A = manager.arange(0, 20, 1, DataType.FLOAT32).reshape(5, 4);
        //B = A.duplicate();
        //println("A", A);
        //println("A + B", A.add(B));
        //// 矩阵的按元素乘法称为哈达玛积
        //a = manager.create(2);
        //NDArray X = manager.arange(24).reshape(2, 3, 4);
        //println("a + X", a.add(X));
        //// 计算其元素的和
        //x = manager.arange(0, 4, 1, DataType.FLOAT32);
        //println("x", x);
        //println("x sum", x.sum());
        //// 表示任意形状张量的元素和，以及按维度计算元素和（累加到某一维）
        //A = manager.arange(20 * 2).reshape(2, 5, 4);
        //println("A", A);
        //println("A shape: " + A.getShape());
        //println("A sum:" + A.sum());
        //NDArray tempNDArray = A.sum(new int[]{0});
        //println("A sum(0) row, shape:" + tempNDArray.getShape(), tempNDArray);
        //tempNDArray = A.sum(new int[]{1});
        //println("A sum(1) column, shape:" + tempNDArray.getShape(), tempNDArray);
        //// 一个与求和相关的量是平均值
        //A = manager.arange(0, 20, 1, DataType.FLOAT32).reshape(5, 4);
        //println("A", A);
        //println("A mean: ", A.mean());
        //println("A sum / numel: " + (A.sum().getFloat() / A.size()));
        //println("A mean axis 0:" + A.mean(new int[]{0}));
        //println("A sum / numel axis 0:" + (A.sum(new int[]{0}).div(A.getShape().size(0))));
        //// 计算总和或均值时保持轴数不变
        //NDArray sum_A = A.sum(new int[]{1}, true);
        //println("sum A", sum_A);
        //// 通过广播将 A 除以 sum_A
        //println("A / sum_A", A.div(sum_A));
        //// 某个轴计算 A 元素的累积总和
        //println("A cumsum", A.cumSum(0));
        //// 点积是相同位置的按元素乘积的和
        //x = manager.arange(0, 4, 1, DataType.FLOAT32);
        //y = manager.ones(new Shape(4));
        //println("x dot y", x.dot(y));
        //// 我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积
        //println("x dot y by * and sum", x.mul(y).sum());
        //// 矩阵向量积，矩阵列数需要与向量长度相同，这样才能累加乘积
        //println("A shape:" + A.getShape(), A);
        //println("x shape:" + x.getShape(), x);
        //println("A mv x", A.dot(x));
        //// 我们可以将矩阵-矩阵乘法看作是简单地执行 m 次矩阵-向量积，并将结果拼接在一起，形成一个 n * m 矩阵
        //B = manager.ones(new Shape(4, 3));
        //println("A mm B", A.matMul(B));
        //// L2 范数是向量元素平方和的平方根
        //NDArray u = manager.create(new float[]{3.f, -4.f});
        //println("u l2 norm", u.norm());
        //// L1 范数，它表示为向量元素的绝对值之和
        //println("u l1 norm(abs -> sum)", u.abs().sum());
        //// 矩阵的 弗罗贝尼乌斯范数（F 范数）是矩阵元素的平方和的平方根
        //println("f norm", manager.ones(new Shape(4, 9)).norm());
            /*
            线性回归
             */
        NDArray trueW = manager.create(new float[]{2, -3.4f});
        float trueB = 4.2f;
        NDList tempNDList = syntheticData(trueW, trueB, 1000);
        NDArray features = tempNDList.get(0);
        NDArray labels = tempNDList.get(1);
        int batchSize = 10;
        NDList[] ndLists = dataIter(batchSize, features, labels);
        // 初始化模型参数
        NDArray w = manager.randomNormal(0f, 0.01f, new Shape(2, 1), DataType.FLOAT32);
        w.setRequiresGradient(true);
        NDArray b = manager.zeros(new Shape(1));
        b.setRequiresGradient(true);
        println("w", w);
        println("b", b);
    }

    /*
    生成数据集
     */
    static NDList syntheticData(NDArray w, float b, int numExamples) {
        //生成 y = Xw + b + 噪音
        NDArray X = manager.randomNormal(0.f, 1.f, new Shape(1000, w.size()), DataType.FLOAT32);
        NDArray y = X.mul(w).add(b);
        y.addi(manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32));
        return new NDList(X, y.reshape(-1, 1));
    }

    /**
     * 读取数据集
     * @param batchSize 批量⼤⼩
     * @param features 特征矩阵
     * @param labels 标签向量
     * @return ⽣成⼤⼩为batch_size的⼩批量, 每个⼩批量包含⼀组特征和标签
     */
    static NDList[] dataIter(int batchSize, NDArray features, NDArray labels) {
        long numExamples = features.getShape().get(0);
        NDArray indices = manager.randomPermutation(numExamples);
        NDList featuresIndices = new NDList();
        NDList labelsIndices = new NDList();
        for (long i = 0; i < numExamples / batchSize + (numExamples % batchSize == 0 ? 0 : 1);) {
            featuresIndices.add(features.get(indices.get(i * batchSize + ":" + (++i * batchSize))));
            labelsIndices.add(labels.get(indices.get(i * batchSize + ":" + (++i * batchSize))));
        }
        return new NDList[]{featuresIndices, labelsIndices};
    }

    /**
     * 定义模型
     * @param X 输⼊特征
     * @param w 模型权重
     * @param b 偏置
     */
    static NDArray linreg(NDArray X, NDArray w, int b) {
        // 线性回归模型
        return X.matMul(w).add(b);
    }

    /**
     * 定义损失函数
     * @param yHat 预测值
     * @param y 真实值
     */
    static NDArray squaredLoss(NDArray yHat, NDArray y) {
        // 均方损失
        return yHat.sub(y.reshape(yHat.getShape())).pow(2).sub(2);
    }

    /**
     * 定义优化算法
     * @param params 受模型参数集合
     * @param lr 学习速率
     * @param batchSize 批量⼤⼩
     */
    static void sgd(NDList params, int lr, int batchSize) {
        //小批量随机梯度下降
        for (NDArray param : params) {
            param.setRequiresGradient(false);
            NDArray p = param.getGradient().mul(lr).div(batchSize);
            param.subi(p);
            param.getGradient().close();
        }
    }

    @Test
    public void run() {
        NDList ndArrays = new NDList(manager.arange(10).split(2));
        System.out.println(Arrays.toString(ndArrays.getShapes()));
    }
}
