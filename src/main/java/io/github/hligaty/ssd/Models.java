package io.github.hligaty.ssd;

import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.object_detection.ssd.SsdBlockFactory;
import ai.djl.modality.cv.MultiBoxDetection;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Models {
    public static final String MODEL_NAME = "pikachu-ssd";
    public static final Path MODEL_PATH = Paths.get("ssd-coco");
    public static final int IMAGE_SIZE = 256;

    public static final int NUM_CLASSES;

    static {
        System.setProperty("ai.djl.default_engine", "MXNet");
        try {
            NUM_CLASSES = Training.getVocDataset().getClasses().size();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private Models() {
    }
    
    public static Model getModel() {
        Model model = Model.newInstance(MODEL_NAME);

        Map<String, Object> arguments = new HashMap<>();
        arguments.put("outSize", NUM_CLASSES);
        arguments.put("numFeatures", 3);
        arguments.put("globalPool", true);
        arguments.put("numFilters", Arrays.asList(16., 32., 64.));
        arguments.put("ratios", Arrays.asList(1., 2., .5));
        arguments.put("sizes", Arrays.asList(
                Arrays.asList(0.2, 0.272),
                Arrays.asList(0.37, 0.447),
                Arrays.asList(0.54, 0.619),
                Arrays.asList(0.71, 0.79),
                Arrays.asList(0.88, 0.961)
        ));
        Block block = new SsdBlockFactory().newBlock(model, MODEL_PATH, arguments);
        model.setBlock(block);
        return model;
    }

    public static void addPredictBlock(Model model) {
        SequentialBlock block = new SequentialBlock();
        Block trainBlock = model.getBlock();
        block.add(trainBlock);
        block.add(new LambdaBlock(output -> {
            // 锚点
            NDArray anchors = output.get(0);
            // 类别
            NDArray classProbabilities = output.get(1).softmax(-1).transpose(0, 2, 1);
            // 边界框
            NDArray boxPredictions = output.get(2);
            MultiBoxDetection multiBoxDetection = MultiBoxDetection.builder().build();
            NDList detections = multiBoxDetection.detection(new NDList(classProbabilities, boxPredictions, anchors));
            return detections.singletonOrThrow().split(new long[]{1, 2}, 2);
        }));
        model.setBlock(block);
    }

    public static void saveSynset(Path modelDir, List<String> synset) throws IOException {
        Files.createDirectories(modelDir);
        Path synsetFile = modelDir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(synsetFile)) {
            writer.write(String.join("\n", synset));
        }
    }
}
