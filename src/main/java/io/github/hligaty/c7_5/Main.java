package io.github.hligaty.c7_5;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;

/**
 * 应用 BatchNorm 于 LeNet 模型
 */
public class Main {
    public static void main(String[] args) {
        SequentialBlock block = new SequentialBlock();

        block
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(5, 5))
                        .setFilters(6)
                        .build())
                .add(BatchNorm.builder().build())
                .add(Activation::sigmoid)
                .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(5, 5))
                        .setFilters(16)
                        .build())
                .add(BatchNorm.builder().build())
                .add(Activation::sigmoid)
                .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(120).build())
                .add(BatchNorm.builder().build())
                .add(Activation::sigmoid)
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(84).build())
                .add(BatchNorm.builder().build())
                .add(Activation::sigmoid)
                .add(Linear.builder().setUnits(10).build());
    }
}
