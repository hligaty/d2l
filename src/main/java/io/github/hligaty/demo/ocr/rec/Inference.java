package io.github.hligaty.demo.ocr.rec;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;

/*
未完待续...
 */
public class Inference {

    public static void run(String imageFilePath) throws IOException, MalformedModelException, TranslateException, ModelNotFoundException {
        try (ZooModel<Image, String> model = RecModel.getModel();
             Predictor<Image, String> predictor = model.newPredictor()) {
            Image image = new BufferedImageFactory().fromFile(Paths.get(imageFilePath));
            System.out.println("ocr_rec.png:\n" + predictor.predict(image));
        }
    }
}
