package io.github.hligaty.demo.det.yolov5;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Inference {

    public static void run(String imageFilePath) throws IOException, MalformedModelException, TranslateException, ModelNotFoundException {
        try (ZooModel<Image, DetectedObjects> model = YoloV5Model.getModel();
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            Image image = ImageFactory.getInstance().fromFile(Paths.get(imageFilePath));
            DetectedObjects result = predictor.predict(image);
            image.drawBoundingBoxes(result);
            image.save(Files.newOutputStream(Paths.get("cat-dog-output.png")), "png");
        }
    }
}
