package io.github.hligaty.ssd;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import static io.github.hligaty.ssd.Models.MODEL_NAME;
import static io.github.hligaty.ssd.Models.MODEL_PATH;


public class Inference {
    // 预测阈值
    public static final float threshold = .6f;

    public static void main(String[] args) throws IOException, MalformedModelException, TranslateException {
        String imageFilePath = "D:\\Repository\\Idea\\d2l\\pikachu.jpg";

        try (Model model = Models.getModel()) {
            model.load(MODEL_PATH, MODEL_NAME);

            Models.addPredictBlock(model);

            SingleShotDetectionTranslator translator = SingleShotDetectionTranslator.builder()
                    .addTransform(new ToTensor())
                    .optThreshold(threshold)
                    .build();

            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor(translator)) {
                Image image = ImageFactory.getInstance().fromFile(Paths.get(imageFilePath));
                DetectedObjects result = predictor.predict(image);
                image.drawBoundingBoxes(result);
                image.save(Files.newOutputStream(Paths.get("pikachu-output.png")), "png");
            }
        }
    }
}
