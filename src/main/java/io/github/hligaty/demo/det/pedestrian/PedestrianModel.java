package io.github.hligaty.demo.det.pedestrian;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import io.github.hligaty.demo.det.utils.PaddleDetectionTranslator;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Collections;

/*
行人检测模型
https://www.paddlepaddle.org.cn/hubdetail?name=yolov3_darknet53_pedestrian&en_category=ObjectDetection
 */
public class PedestrianModel {

    public static ZooModel<Image, DetectedObjects> getModel() throws ModelNotFoundException, MalformedModelException, IOException {
        PaddleDetectionTranslator pedestrianTranslator = PaddleDetectionTranslator.builder()
                .optImageSize(608f, 608f)
                .optClasses(Collections.singletonList("pedestrian"))
                .optThreshold(.5f)
                .addInputParam(PaddleDetectionTranslator.PaddleDetectionInputType.IM_SHAPE)
                .addInputParam(PaddleDetectionTranslator.PaddleDetectionInputType.IMAGE)
                .addInputParam(PaddleDetectionTranslator.PaddleDetectionInputType.SCALE_FACTOR)
                .build();
        return Criteria.builder()
                .optEngine("OnnxRuntime")
                .setTypes(Image.class, DetectedObjects.class)
                .optModelPath(Paths.get("src/main/resources/pedestrian_yolov3_darknet.onnx"))
                .optTranslator(pedestrianTranslator)
                .build().loadModel();
    }
}
