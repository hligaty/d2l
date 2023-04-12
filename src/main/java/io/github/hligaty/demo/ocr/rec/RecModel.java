package io.github.hligaty.demo.ocr.rec;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.paddlepaddle.zoo.cv.wordrecognition.PpWordRecognitionTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;

import java.io.IOException;
import java.nio.file.Paths;

public class RecModel {
    public static ZooModel<Image, String> getModel() throws ModelNotFoundException, MalformedModelException, IOException {
        return Criteria.builder()
                /*
                这个模型转 ONNX 没有报错, 但运行会报错
                另外 PaddlePaddle 的模型在多线程下必须对应多个模型, 将其转为其他模型更好
                 */
                .optEngine("PaddlePaddle")
                .setTypes(Image.class, String.class)
                .optModelPath(Paths.get("src/main/resources/ch_PP-OCRv3_rec_infer"))
                .optTranslator(new PpWordRecognitionTranslator())
                .build().loadModel();
    }
}
