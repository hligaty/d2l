package io.github.hligaty.c3_2;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;

import java.io.IOException;

import static io.github.hligaty.c3_2.Training.*;

public class Main {
    
    /*
    1. 通过 NDArray 进行数据存储和线性代数
    2. 通过 GradientCollector 计算梯度
     */
    public static void main(String[] args) throws TranslateException, IOException {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "MXNet")) {
            NDArray trueW = manager.create(new float[]{2, -3.4f});
            float trueB = 4.2f;
            // 生成数据集
            DataPoints dp = DataPoints.syntheticData(manager, trueW, trueB, 1000);
            NDArray features = dp.getX();
            NDArray labels = dp.getY();
            // 读取数据集
            int batchSize = 10;
            ArrayDataset dataset = new ArrayDataset.Builder()
                    .setData(features)
                    .optLabels(labels)
                    .setSampling(batchSize, false)
                    .build();
            // 初始化权重，并将偏置初始化为0
            NDArray w = manager.randomNormal(0, 0.01f, new Shape(2, 1), DataType.FLOAT32);
            NDArray b = manager.zeros(new Shape(1));
            NDList params = new NDList(w, b);
            /*
            初始化之后，我们的任务是更新这些参数，知道这些参数足够拟合我们的数据。
            每次更新都需要计算损失函数关于模型参数的梯度。然后按照反梯度方向更新每个参数。
             */
            // 训练
            /*
            学习率
            不能太小（每次迭代损失降低的太小，虽然可以多迭代几次）
            不能太大（loss NaN，即 loss not a number，求导可能除 0 或 无限，超出浮点运算范围了）
             */
            float lr = 0.03f;
            int numEpochs = 3; // 迭代周期个数
            // Attach Gradients
            for (NDArray param : params) {
                param.setRequiresGradient(true);
            }
            for (int epoch = 0; epoch < numEpochs; epoch++) {
                // X 是特征，y 是标签
                for (Batch batch : dataset.getData(manager)) {
                    NDArray X = batch.getData().head();
                    NDArray y = batch.getLabels().head();
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        // X 和 y 的小批量损失
                        NDArray l = squaredLoss(linreg(X, params.get(0), params.get(1)), y);
                        // 计算 l 关于 w 和 b 的梯度
                        gc.backward(l);
                    }
                    // 使用梯度进行更新
                    sgd(params, lr, batchSize);
                    batch.close();
                }
                // 把整个 features 传进去计算预测，在和真实的 labels 做一下损失
                // 可以看到 loss 每过一轮都在变小
                NDArray trainL = squaredLoss(linreg(features, params.get(0), params.get(1)), labels);
                System.out.printf("epoch %d, loss %f\n", epoch + 1, trainL.mean().getFloat());
            }
            float[] wError = trueW.sub(params.get(0).reshape(trueW.getShape())).toFloatArray();
            System.out.printf("误差 w: [%f, %f]%n", wError[0], wError[1]);
            System.out.printf("误差 b: %f%n", trueB - params.get(1).getFloat());
        }
    }
}
