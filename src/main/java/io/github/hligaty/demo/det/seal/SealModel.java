package io.github.hligaty.demo.det.seal;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import io.github.hligaty.demo.det.utils.PaddleDetectionTranslator;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

/*
印章模型
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/applications/%E5%8D%B0%E7%AB%A0%E5%BC%AF%E6%9B%B2%E6%96%87%E5%AD%97%E8%AF%86%E5%88%AB.md#%E5%8D%B0%E7%AB%A0%E5%BC%AF%E6%9B%B2%E6%96%87%E5%AD%97%E8%AF%86%E5%88%AB
 */
public class SealModel {

    public static ZooModel<Image, DetectedObjects> getModel() throws ModelNotFoundException, MalformedModelException, IOException {
        PaddleDetectionTranslator sealTranslator = PaddleDetectionTranslator.builder()
                .optImageSize(320f, 320f)
                .optClasses(Arrays.asList("seal", "seal"))
                .addInputParam(PaddleDetectionTranslator.PaddleDetectionInputType.IM_SHAPE)
                .addInputParam(PaddleDetectionTranslator.PaddleDetectionInputType.IMAGE)
                .addInputParam(PaddleDetectionTranslator.PaddleDetectionInputType.SCALE_FACTOR)
                .build();
        return Criteria.builder()
                .optEngine("OnnxRuntime")
                .setTypes(Image.class, DetectedObjects.class)
                .optModelPath(Paths.get("src/main/resources/seal_det_ppyolo.onnx"))
                .optTranslator(sealTranslator)
                .build().loadModel();
    }
}
