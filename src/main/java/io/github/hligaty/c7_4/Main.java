package io.github.hligaty.c7_4;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * GoogLeNet
 */
public class Main {
    public static void main(String[] args) {
        SequentialBlock block1 = new SequentialBlock();
        block1
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(7, 7))
                        .optStride(new Shape(2, 2))
                        .optPadding(new Shape(3, 3))
                        .setFilters(64)
                        .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        SequentialBlock block2 = new SequentialBlock();
        block2
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .setFilters(192)
                        .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        SequentialBlock block3 = new SequentialBlock();
        block3
                .add(inceptionBlock(64, new int[]{96, 128}, new int[]{16, 32}, 32))
                .add(inceptionBlock(128, new int[]{128, 192}, new int[]{32, 96}, 64))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        SequentialBlock block4 = new SequentialBlock();
        block4
                .add(inceptionBlock(192, new int[]{96, 208}, new int[]{16, 48}, 64))
                .add(inceptionBlock(160, new int[]{112, 224}, new int[]{24, 64}, 64))
                .add(inceptionBlock(128, new int[]{128, 256}, new int[]{24, 64}, 64))
                .add(inceptionBlock(112, new int[]{144, 288}, new int[]{32, 64}, 64))
                .add(inceptionBlock(256, new int[]{160, 320}, new int[]{32, 128}, 128))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        SequentialBlock block5 = new SequentialBlock();
        block5
                .add(inceptionBlock(256, new int[]{160, 320}, new int[]{32, 128}, 128))
                .add(inceptionBlock(384, new int[]{192, 384}, new int[]{48, 128}, 128))
                .add(Pool.globalAvgPool2dBlock());

        SequentialBlock block = new SequentialBlock();
        block.addAll(block1, block2, block3, block4, block5, Linear.builder().setUnits(10).build());
    }

    static ParallelBlock inceptionBlock(int c1, int[] c2, int[] c3, int c4) {
        SequentialBlock p1 = new SequentialBlock();
        p1
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(1, 1))
                        .setFilters(c1)
                        .build())
                .add(Activation::relu);

        SequentialBlock p2 = new SequentialBlock();
        p2
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(1, 1))
                        .setFilters(c2[0])
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optPadding(new Shape(1, 1))
                        .setFilters(c2[1])
                        .build())
                .add(Activation::relu);

        SequentialBlock p3 = new SequentialBlock();
        p3
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(1, 1))
                        .setFilters(c3[0])
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(5, 5))
                        .optPadding(new Shape(2, 2))
                        .setFilters(c3[1])
                        .build())
                .add(Activation::relu);

        SequentialBlock p4 = new SequentialBlock();
        p4
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(1, 1), new Shape(1, 1)))
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(1, 1))
                        .setFilters(c4)
                        .build())
                .add(Activation::relu);

        return new ParallelBlock(ndLists -> {
            List<NDArray> list = ndLists.stream()
                    .map(NDList::head)
                    .collect(Collectors.toList());
            return new NDList(NDArrays.concat(new NDList(list), 1));
        }, Arrays.asList(p1, p2, p3, p4));
    }
}
