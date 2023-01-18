package io.github.hligaty.c3_3;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.L2Loss;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.Sgd;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import io.github.hligaty.c3_2.DataPoints;
import io.github.hligaty.utils.Util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Main {
    /*
    实际上，由于数据迭代器、损失函数、优化器和神经网络层很常用，现代深度学习库也为我们实现了这些组件.
    ai.djl.training.dataset 下有很多数据处理工具
    ai.djl.nn 下有定义了大量的神经网络构建层 layer
    Loss 类定义了许多通用的损失函数
    ai.djl.training.initializer 有各种模型初始化方法及相应的初始化配置参数供参考
     */
    public static void main(String[] args) throws TranslateException, IOException {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "MXNet")) {
            // 还是生成一下数据
            NDArray trueW = manager.create(new float[]{2, -3.4f});
            float trueB = 4.2f;

            DataPoints dp = DataPoints.syntheticData(manager, trueW, trueB, 1000);
            NDArray features = dp.getX();
            NDArray labels = dp.getY();

            // 读取数据集
            int batchSize = 10;
            ArrayDataset dataset = Util.loadArray(features, labels, batchSize, false);

            // 定义模型，只需要关注使用哪些层来构造模型，不必关注层的实现细节
            try (Model model = Model.newInstance("lin-reg")) {
                /*
                SequentialBlock 类为串联在一起的多个层定义一个容器。
                当给定输入数据，SequentialBlock 实例将数据传入到第一层，因此实际上不需要 SequentialBlock，
                但由于以后几乎所有的模型都是多层的，这里使用 SequentialBlock 会让你熟悉标准的流水线
                 */
                SequentialBlock net = new SequentialBlock();
                /*
                全连接层即每一个输入都通过矩阵-向量乘法连接到它的每个输出
                定义全连接层，只想得到一个标量输出，因此设置维度为 1，并设置包含偏差
                 */
                Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build();
                net.add(linearBlock);
                model.setBlock(net);
                /*
                定义损失函数
                使用平方损失(L2Loss)
                 */
                L2Loss l2Loss = Loss.l2Loss();
                /*
                定义优化算法
                小批量随机梯度下降算法是一种优化神经网络的标准工具，DJL 通过 Optimizer 类支持该算法的许多
                变种。当我们实例化 Optimizer 时，我们要指定优化的参数。即希望使用的优化算法（sgd）以及
                优化算法所需的超参数字典。小批量孙吉梯度下降只需要设置学习率，这里设为 0.03
                 */
                Tracker lrt = Tracker.fixed(0.03f);
                Sgd sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();
                /*
                Trainer的初始化配置
                初始化以及配置 trainer，并用这个 trainer 对人工智能模型进行训练
                 */
                DefaultTrainingConfig config = new DefaultTrainingConfig(l2Loss)
                        .optOptimizer(sgd) // 优化函数
                        .optDevices(manager.getEngine().getDevices()) // 设备
                        .addTrainingListeners(TrainingListener.Defaults.logging());// 日志打印

                Trainer trainer = model.newTrainer(config);
                /*
                初始化模型参数
                对人工智能模型进行训练前，需要对模型的参数进行初始化设置。
                例如：对线性回归模型进行初始化配置时，我们需要提供权重及偏差参数，
                再通过 initialize 传入即可
                 */
                // 第一个轴是批量大小，不会影响参数初始化
                // 第二个轴是输入大小
                trainer.initialize(new Shape(batchSize, 2));
                /*
                运行性能指标
                一般情况，DJL 不会自动记录运行性能指标，因为记录运行指标本身会提高运行成本，降低性能。
                如果有特殊理由，需要记录，那么可以生成一个 metrics 并设置到 trainer
                 */
                Metrics metrics = new Metrics();
                trainer.setMetrics(metrics);
                /*
                训练
                 */
                int numEpochs = 3;
                for (int epoch = 1; epoch <= numEpochs; epoch++) {
                    System.out.printf("Epoch %d\n", epoch);
                    // 迭代数据集
                    for (Batch batch : trainer.iterateDataset(dataset)) {
                        // 小批量随机梯度下降
                        // 计算 loss
                        EasyTrain.trainBatch(trainer, batch);
                        // 更新参数
                        trainer.step();
                        batch.close();
                    }
                    // 迭代后重置
                    trainer.notifyListeners(trainingListener -> trainingListener.onEpoch(trainer));
                }
                /*
                将真实原始参数（权重 trueW 和偏差 trueB）和模型训练中学习到的参数（wParam 和 bParam）进行比较。
                在 DJL 里，访问模型训练中学习到的参数需要分两步走。
                1：从模型 model 中取出构建层 layer，从构建层中用 getParamters() 函数取参数列表。
                2：取得参数列表后，每个独立的参数就可以通过 valueAt() 函数用列表下标获取了。
                下面获取的参数 0 是权重，1 是 偏差参数
                 */
                Block layer = model.getBlock();
                ParameterList params = layer.getParameters();
                NDArray wParam = params.valueAt(0).getArray();
                NDArray bParam = params.valueAt(1).getArray();
                float[] w = trueW.sub(wParam.reshape(trueW.getShape())).toFloatArray();
                System.out.printf("Error in estimating w: [%f %f]\n", w[0], w[1]);
                System.out.printf("Error in estimating b: %f\n%n", trueB - bParam.getFloat());
                /*
                保存训练模型
                还应该保存模型的元数据。
                 */
                Path modelDir = Paths.get("../models/lin-reg");
                Files.createDirectories(modelDir);

                model.setProperty("Epoch", Integer.toString(numEpochs));
                model.save(modelDir, "lin-reg");
                System.out.println(model);
            }
        }
    }
}
