package io.github.hligaty.c3_6;


import ai.djl.Device;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import io.github.hligaty.c3_2.Training;
import io.github.hligaty.utils.Accumulator;
import io.github.hligaty.utils.FashionMnistUtils;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

public class Main {
    static NDManager manager;
    static NDList params;
    static int numInputs = 784;
    static int numOutPuts = 10;

    static int batchSize = 256;
    static boolean randomShuffle = true;
    static int numEpochs = 5;
    static float lr = 0.1f;

    /*
    初始化模型参数
    输入是 28 * 28 的图像，展平图像，将它们看作为长度784的向量。后面将讨论能够利用图像
    空间结构的更为复杂的策略，但现在我们暂时只把每个像素位置看作一个特征。
    回想一下，在 softmax 回归中，我们的输出与类别一样多。因为我们的数据集有 10 个类别，
    所以网络输出维度为 10.因此，权重将构成一个 784 * 10 的矩阵，偏置将构成一个 1 * 10
    的行向量。与线性回归一样，我们将使用正态分布初始化我们的权重 W，偏置初始化为 0.
     */
    public static void main(String[] args) throws TranslateException, IOException {
        // 获取训练集和验证集
        FashionMnist trainingSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();
        FashionMnist validationSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, false)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();
        try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "MXNet")) {
            Main.manager = manager;
            NDArray W = manager.randomNormal(0, 0.01f, new Shape(numInputs, numOutPuts), DataType.FLOAT32);
            NDArray b = manager.zeros(new Shape(numOutPuts), DataType.FLOAT32);
            params = new NDList(W, b);
            trainCh3(Net::net, trainingSet, validationSet, LossFunction::crossEntropy, numEpochs, Updater::updater);
            /*
            预测
             */
            predictCh3(Net::net, validationSet, 6, manager);
        }
    }

    /*
    定义 softmax 操作
    我们将任何随机输入的每个元素都变成一个非负数。
    此外，根据概率原理，每行总和为 1。
    注意，虽然这在数学上看起来是正确的，但我们在代码实现中有点草率。
    矩阵中的非常大或非常小的元素可能导致数值上溢和下溢，但我们没有采取措施来防止这点
     */
    static NDArray softmax(NDArray X) {
        NDArray Xexp = X.exp();
        NDArray partition = Xexp.sum(new int[]{1}, true);
        return Xexp.div(partition); // 这里用了广播机制
    }

    /*
    实现 softmax 回归模型
     */
    public static class Net {
        public static NDArray net(NDArray X) {
            NDArray currentW = params.get(0);
            NDArray currentB = params.get(1);
            // reshape(-1, numInputs) 即 numInputs 列，自适应行
            return softmax(X.reshape(new Shape(-1, numInputs)).dot(currentW).add(currentB));
        }
    }

    /*
    交叉熵损失函数
     */
    static class LossFunction {
        public static NDArray crossEntropy(NDArray yHat, NDArray y) {
            // ???，:,{}获取的是一个 m * n 的矩阵，并不是 m 个元素的向量啊！！！
            return yHat.get(new NDIndex(":, {}", y.toType(DataType.INT32, false))).log().neg();
        }
    }

    /*
    计算正确预测的数量
     */
    static float accuracy(NDArray yHat, NDArray y) {
        return (yHat.getShape().size(1) > 1) ?
                yHat.argMax(1).toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false))
                        .sum().toType(DataType.FLOAT32, false).getFloat() :
                yHat.toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false))
                        .sum().toType(DataType.FLOAT32, false).getFloat();
    }

    /*
    计算准确率
     */
    static float evaluateAccuracy(UnaryOperator<NDArray> net, Iterable<Batch> dataIterator) {
        Accumulator metric = new Accumulator(2);
        for (Batch batch : dataIterator) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();
            // X 放到 net 中先计算预测值，再 accuracy 得到正确预测数，把正确预测数和预测总数放到累加器里
            metric.add(new float[]{accuracy(net.apply(X), y), y.size()});
            batch.close();
        }
        // 计算当前的准确率
        return metric.get(0) / metric.get(1);
    }

    @FunctionalInterface
    public interface ParamConsumer {
        void accept(NDList params, float lr, int batchSize);
    }

    /*
    训练一次
     */
    static float[] trainEpochCh3(UnaryOperator<NDArray> net, Iterable<Batch> trainIter, BinaryOperator<NDArray> loss, ParamConsumer updater) {
        Accumulator metric = new Accumulator(3);

        for (NDArray param : params) {
            param.setRequiresGradient(true);
        }

        for (Batch batch : trainIter) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();
            X.reshape(new Shape(-1, numInputs));

            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray yHat = net.apply(X); // 计算预测值
                NDArray l = loss.apply(yHat, y); // 交叉熵损失函数计算 l
                gc.backward(l); // 计算 l 关于 w 和 b 的梯度
                metric.add(new float[]{
                        l.sum().toType(DataType.FLOAT32, false).getFloat(), // 损失的累加
                        accuracy(yHat, y), // 正确分类样本数
                        y.size()}); // 样本总数
            }
            updater.accept(params, lr, batch.getSize());
            batch.close();
        }
        return new float[]{metric.get(0) / metric.get(2), metric.get(1) / metric.get(2)};
    }

    static void trainCh3(UnaryOperator<NDArray> net, Dataset trainDataset, Dataset testDataset,
                         BinaryOperator<NDArray> loss, int numEpochs, ParamConsumer updater) throws TranslateException, IOException {
        //Animator animator = new Animator();
        for (int i = 0; i < numEpochs; i++) {
            // 训练一次并更新模型，并拿到训练的误差
            float[] trainMetrics = trainEpochCh3(net, trainDataset.getData(manager), loss, updater);
            // 在测试数据集上评估精度
            float accuracy = evaluateAccuracy(net, testDataset.getData(manager));
            float trainAccuracy = trainMetrics[0];
            float trainLoss = trainMetrics[1];

            //animator.add(i, accuracy, trainAccuracy, trainLoss);
            System.out.printf("Epoch %d: Test Accuracy: %f\n", i, accuracy);
            System.out.printf("Train Accuracy: %f\n", trainAccuracy);
            System.out.printf("Train Loss: %f\n", trainLoss);
        }
    }

    /*
    优化算法
     */
    static class Updater {
        public static void updater(NDList params, float lr, int batchSize) {
            Training.sgd(params, lr, batchSize);
        }
    }

    static BufferedImage predictCh3(UnaryOperator<NDArray> net, ArrayDataset dataset, int number, NDManager manager) throws TranslateException, IOException {
        final int SCALE = 4;
        final int WIDTH = 28;
        final int HEIGHT = 28;
        
        int[] predLables = new int[number];

        for (Batch batch : dataset.getData(manager)) {
            NDArray X = batch.getData().head();
            int[] yHat = net.apply(X).argMax(1).toType(DataType.INT32, false).toIntArray();
            for (int i = 0; i < number; i++) {
                predLables[i] = yHat[i];
            }
            break;
        }
        return FashionMnistUtils.showImages(dataset, predLables, WIDTH, HEIGHT, SCALE, manager);
    }
}
