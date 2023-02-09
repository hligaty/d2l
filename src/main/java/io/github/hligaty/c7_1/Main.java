package io.github.hligaty.c7_1;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;

/**
 * AlexNet
 */
public class Main {
    public static void main(String[] args) {
        SequentialBlock block = new SequentialBlock();

        block
                // 先来两套卷积激活池化
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(11, 11))
                        .optStride(new Shape(4, 4))
                        .optPadding(new Shape(1, 1))
                        .setFilters(96)
                        .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(5, 5))
                        .optPadding(new Shape(2, 2))
                        .setFilters(256)
                        .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // 再来三台卷积激活，最后池化
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .setFilters(384)
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .setFilters(384)
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .setFilters(256)
                        .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // 然后展开来个多层感知机，两套隐藏层（全连接）
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder()
                        .setUnits(4096)
                        .build())
                .add(Activation::relu)
                .add(Dropout.builder()
                        .optRate(.5f)
                        .build())
                .add(Linear.builder()
                        .setUnits(4096)
                        .build())
                .add(Activation::relu)
                .add(Dropout.builder()
                        .optRate(.5f)
                        .build())
                // 最后来个全连接层输出类别
                .add(Linear.builder()
                        .setUnits(10)
                        .build());
    }
}
