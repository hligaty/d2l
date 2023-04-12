package io.github.hligaty.demo.ocr.det;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.paddlepaddle.zoo.cv.objectdetection.PpWordDetectionTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.concurrent.ConcurrentHashMap;

public class DetModel {
    public static ZooModel<Image, DetectedObjects> getModel() throws ModelNotFoundException, MalformedModelException, IOException {
        return Criteria.builder()
                .optEngine("OnnxRuntime")
                .setTypes(Image.class, DetectedObjects.class)
                .optModelPath(Paths.get("src/main/resources/ch_pp_ocrv3_det.onnx"))
                .optTranslator(new PpWordDetectionTranslator(new ConcurrentHashMap<>()))
                .build().loadModel();
    }
}
