package io.github.hligaty.demo.det.utils;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PaddleDetectionTranslator implements NoBatchifyTranslator<Image, DetectedObjects> {
    private final float threshold;
    private final float imageWidth;
    private final float imageHeight;
    private final List<String> classes;
    private final List<PaddleDetectionInputType> inputTypes;
    private final Pipeline pipeline;
    private float[] scaleFactor;

    private PaddleDetectionTranslator(Builder builder) {
        threshold = builder.threshold;
        imageWidth = builder.imageWidth;
        imageHeight = builder.imageHeight;
        classes = builder.classes;
        inputTypes = builder.inputTypes;
        pipeline = new Pipeline();
        pipeline.add(new Resize((int) imageWidth, (int) imageHeight))
                .add(new ToTensor())
                .add(new Normalize(new float[]{0.485f, 0.456f, 0.406f}, new float[]{0.229f, 0.224f, 0.225f}))
                .add(array -> array.expandDims(0));
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDManager manager = ctx.getNDManager();
        NDList ndList = pipeline.transform(new NDList(input.toNDArray(manager, Image.Flag.COLOR)));
        NDArray image = ndList.remove(0);
        for (PaddleDetectionInputType inputType : inputTypes) {
            switch (inputType) {
                case IMAGE:
                    ndList.add(image);
                    break;
                case IM_SHAPE:
                    ndList.add(manager.create(new float[]{input.getHeight(), input.getWidth()}).expandDims(0));
                    break;
                case SCALE_FACTOR:
                    ndList.add(manager.create(new float[]{input.getHeight() / imageHeight, input.getWidth() / imageWidth}).expandDims(0));
                    break;
                case SCALE_FACTOR_NORMAL:
                    ndList.add(manager.create(scaleFactor = new float[]{input.getHeight() / imageHeight, input.getWidth() / imageWidth}).expandDims(0));
                    break;
                default:
                    throw new IllegalArgumentException("Unknown PaddleDetectionInputType");
            }
        }
        return ndList;
    }

    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        NDArray result = list.get(0);
        int[] classIndices = result.get(":, 0").toType(DataType.INT32, true).flatten().toIntArray();
        double[] probs = result.get(":, 1").toType(DataType.FLOAT64, true).toDoubleArray();
        int detected = Math.toIntExact(probs.length);

        NDArray xMin = result.get(":, 2:3").clip(0, imageWidth).div(imageWidth);
        NDArray yMin = result.get(":, 3:4").clip(0, imageHeight).div(imageHeight);
        NDArray xMax = result.get(":, 4:5").clip(0, imageWidth).div(imageWidth);
        NDArray yMax = result.get(":, 5:6").clip(0, imageHeight).div(imageHeight);

        float[] boxX = xMin.toFloatArray();
        float[] boxY = yMin.toFloatArray();
        float[] boxWidth = xMax.sub(xMin).toFloatArray();
        float[] boxHeight = yMax.sub(yMin).toFloatArray();

        List<String> retClasses = new ArrayList<>(detected);
        List<Double> retProbs = new ArrayList<>(detected);
        List<BoundingBox> retBB = new ArrayList<>(detected);
        for (int i = 0; i < detected; i++) {
            if (classIndices[i] < 0 || probs[i] < threshold) {
                continue;
            }
            retClasses.add(classes.get(0));
            retProbs.add(probs[i]);
            retBB.add(scaleFactor == null ?
                    new Rectangle(boxX[i], boxY[i], boxWidth[i], boxHeight[i]) :
                    new Rectangle(boxX[i] * scaleFactor[1], boxY[i] * scaleFactor[0], boxWidth[i] * scaleFactor[1], boxHeight[i] * scaleFactor[0]));
        }
        return new DetectedObjects(retClasses, retProbs, retBB);
    }

    public static Builder builder() {
        return new Builder();
    }

    /*
    https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.2/deploy/EXPORT_MODEL.md#1%E5%AF%BC%E5%87%BA%E6%A8%A1%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E8%AF%B4%E6%98%8E
     */
    public enum PaddleDetectionInputType {
        IMAGE,
        IM_SHAPE,
        /*
        与上述文档描述一致的
         */
        SCALE_FACTOR_NORMAL,
        /*
        与上述文档描述相反的, 如果输出的结果标错位置了, 用这个试试
         */
        SCALE_FACTOR
    }

    public static class Builder {
        private float threshold = 0.5f;
        private float imageWidth;
        private float imageHeight;
        private List<PaddleDetectionInputType> inputTypes;
        private List<String> classes;

        /**
         * 设置预测阈值
         *
         * @param threshold 低于该值将被过滤, 默认为 0.5
         * @return Builder
         */
        public Builder optThreshold(float threshold) {
            this.threshold = threshold;
            return self();
        }

        /**
         * 设置类别
         *
         * @param classes 类别
         * @return Builder
         */
        public Builder optClasses(List<String> classes) {
            this.classes = classes;
            return self();
        }

        /**
         * 设置模型输入图片的尺寸
         *
         * @param imageWidth  宽度
         * @param imageHeight 高度
         * @return Builder
         */
        public Builder optImageSize(float imageWidth, float imageHeight) {
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            return self();
        }

        /**
         * 添加模型输入参数
         *
         * @param inputType 参数类型
         * @return Builder
         */
        public Builder addInputParam(PaddleDetectionInputType inputType) {
            if (inputTypes == null) {
                inputTypes = new ArrayList<>();
            }
            inputTypes.add(inputType);
            return self();
        }

        /**
         * 添加模型输入参数
         *
         * @param inputTypes 参数类型
         * @return Builder
         */
        public Builder addInputParams(PaddleDetectionInputType... inputTypes) {
            if (this.inputTypes == null) {
                this.inputTypes = new ArrayList<>();
            }
            this.inputTypes.addAll(Arrays.asList(inputTypes));
            return self();
        }

        private Builder self() {
            return this;
        }

        private void validate() {
            if (imageWidth == 0) {
                throw new IllegalArgumentException("imageWidth is required.");
            }
            if (imageHeight == 0) {
                throw new IllegalArgumentException("imageHeight is required.");
            }
            if (classes == null) {
                throw new IllegalArgumentException("classes is required.");
            }
            if (inputTypes == null) {
                addInputParam(PaddleDetectionInputType.IMAGE);
            }
        }

        public PaddleDetectionTranslator build() {
            validate();
            return new PaddleDetectionTranslator(this);
        }
    }
}
