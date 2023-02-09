package io.github.hligaty.c7_3;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;

/**
 * NiN
 */
public class Main {
    public static void main(String[] args) {
        SequentialBlock block = new SequentialBlock();

        block
                .add(niNBlock(96, new Shape(11, 11), new Shape(4, 4), new Shape(0, 0)))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                .add(niNBlock(256, new Shape(5, 5), new Shape(1, 1), new Shape(2, 2)))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                .add(niNBlock(384, new Shape(3, 3), new Shape(1, 1), new Shape(1, 1)))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                .add(Dropout.builder().optRate(.5f).build())
                .add(niNBlock(10, new Shape(3, 3), new Shape(1, 1), new Shape(1, 1)))
                .add(Pool.globalAvgPool2dBlock())
                .add(Blocks.batchFlattenBlock());
    }

    static SequentialBlock niNBlock(int numChannels, Shape kernelShape, Shape strideShape, Shape paddingShape) {
        SequentialBlock block = new SequentialBlock();

        block
                .add(Conv2d.builder()
                        .setKernelShape(kernelShape)
                        .optStride(strideShape)
                        .optPadding(paddingShape)
                        .setFilters(numChannels)
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(1, 1))
                        .setFilters(numChannels)
                        .build()) 
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(1, 1))
                        .setFilters(numChannels)
                        .build())
                .add(Activation::relu);
        return block;
    }
}
