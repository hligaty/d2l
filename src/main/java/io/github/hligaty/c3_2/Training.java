package io.github.hligaty.c3_2;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

public class Training {
    // 定义模型
    public static NDArray linreg(NDArray X, NDArray w, NDArray b) {
        return X.dot(w).add(b);
    }

    // 定义损失函数
    public static NDArray squaredLoss(NDArray yHat, NDArray y) {
        return yHat.sub(y.reshape(yHat.getShape())).mul(
                yHat.sub(y.reshape(yHat.getShape()))
        ).div(2);
    }

    // 定义优化算法
    public static void sgd(NDList params, float lr, int batchSize) {
        for (NDArray param : params) {
            // Update param
            // param = param - param.gradient * lr / batchSize
            param.subi(param.getGradient().mul(lr).div(batchSize));
        }
    }
}
