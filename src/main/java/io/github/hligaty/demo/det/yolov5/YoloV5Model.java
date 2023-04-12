package io.github.hligaty.demo.det.yolov5;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;

import java.io.IOException;
import java.nio.file.Paths;

/*
YOLOV5 检测模型
https://github.com/ultralytics/yolov5
 */
public class YoloV5Model {

    public static ZooModel<Image, DetectedObjects> getModel() throws ModelNotFoundException, MalformedModelException, IOException {
        Translator<Image, DetectedObjects> translator = YoloV5Translator.builder()
                .addTransform(new Resize(640, 640))
                .addTransform(new ToTensor())
                .optSynsetArtifactName("coco.names")
                .optThreshold(.5f)
                .optApplyRatio(true)
                .optRescaleSize(640, 640)
                .build();
        return Criteria.builder()
                .optEngine("OnnxRuntime")
                .optDevice(Device.cpu())
                .setTypes(Image.class, DetectedObjects.class)
                .optModelPath(Paths.get("src/main/resources/yolov5s.onnx"))
                .optTranslator(translator)
                .build().loadModel();
    }
}
