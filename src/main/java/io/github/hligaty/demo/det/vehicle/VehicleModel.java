package io.github.hligaty.demo.det.vehicle;

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
车辆检测
 */
public class VehicleModel {

    public static ZooModel<Image, DetectedObjects> getModel() throws ModelNotFoundException, MalformedModelException, IOException {
        PaddleDetectionTranslator pedestrianTranslator = PaddleDetectionTranslator.builder()
                .optImageSize(640f, 640f)
                .optClasses(Arrays.asList("vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle"))
                .optThreshold(.5f)
                .addInputParam(PaddleDetectionTranslator.PaddleDetectionInputType.IMAGE)
                .addInputParam(PaddleDetectionTranslator.PaddleDetectionInputType.SCALE_FACTOR_NORMAL)
                .build();
        return Criteria.builder()
                .optEngine("OnnxRuntime")
                .setTypes(Image.class, DetectedObjects.class)
                .optModelPath(Paths.get("src/main/resources/mot_ppyoloe_l_36e_ppvehicle.onnx"))
                .optTranslator(pedestrianTranslator)
                .build().loadModel();
    }
}
