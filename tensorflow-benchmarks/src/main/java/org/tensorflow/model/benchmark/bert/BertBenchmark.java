/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.model.benchmark.bert;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.tensorflow.ConcreteFunction;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TInt32;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;

@Fork(value = 1, jvmArgs = {"-Xms4G", "-Xmx4G"})
@BenchmarkMode(Mode.AverageTime)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@State(Scope.Benchmark)
public class BertBenchmark {

    private static final String MODEL_ROOT_DIR = System.getProperty("MODEL_PATH","./models/");

    private static final String BERT_MODEL_PATH = Paths.get(MODEL_ROOT_DIR,"bert").toAbsolutePath().toString();

    private static final String BERT_MODEL_TAG = "serve";

    /* Use tag "serve"
    Signature for "serving_default":
	Method: "tensorflow/serving/predict"
	Inputs:
		"input_word_ids": dtype=DT_INT32, shape=(-1, -1)
		"input_mask": dtype=DT_INT32, shape=(-1, -1)
		"input_type_ids": dtype=DT_INT32, shape=(-1, -1)
	Outputs:
		"bert_encoder": dtype=DT_FLOAT, shape=(-1, 768)
		"bert_encoder_10": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_1": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_11": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_2": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_12": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_3": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_13": dtype=DT_FLOAT, shape=(-1, 768)
		"bert_encoder_4": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_14": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_5": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_6": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_7": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_8": dtype=DT_FLOAT, shape=(-1, -1, 768)
		"bert_encoder_9": dtype=DT_FLOAT, shape=(-1, -1, 768)
     */

    public static void main(String[] args) throws IOException {
        org.openjdk.jmh.Main.main(args);
    }

    @State(Scope.Thread)
    public static class BertModel {
        public SavedModelBundle bert;
        public ConcreteFunction inferenceFunc;

        @Setup(Level.Trial)
        public void loadBert() {
            bert = SavedModelBundle.load(BERT_MODEL_PATH, BERT_MODEL_TAG);
            inferenceFunc = bert.function("serving_default");
        }

        @TearDown(Level.Trial)
        public void closeBert() {
            inferenceFunc.close();
            bert.close();
        }
    }

    @State(Scope.Thread)
    public static class BertInput8x64 {
        public Map<String,Tensor> feedDict;
        @Setup(Level.Trial)
        public void createInput() {
            feedDict = new HashMap<>();
            SplittableRandom rng = new SplittableRandom(42L);
            TInt32 inputWordIds = TInt32.tensorOf(Shape.of(8,64));
            TInt32 inputMask = TInt32.tensorOf(Shape.of(8,64));
            TInt32 inputTypeIds = TInt32.tensorOf(Shape.of(8,64));
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 64; j++) {
                    inputWordIds.setInt(rng.nextInt(30000),i,j);
                    inputMask.setInt(0,i,j);
                    inputTypeIds.setInt(0,i,j);
                }
            }
            feedDict.put("input_word_ids",inputWordIds);
            feedDict.put("input_mask",inputMask);
            feedDict.put("input_type_ids",inputTypeIds);
        }

        @TearDown(Level.Trial)
        public void closeInput() {
            for (Map.Entry<String,Tensor> e : feedDict.entrySet()) {
                e.getValue().close();
            }
        }
    }

    @Benchmark
    @Measurement(batchSize = 1)
    public void loadCloseModel(Blackhole blackhole) {
        try (SavedModelBundle bundle = SavedModelBundle.load(BERT_MODEL_PATH,BERT_MODEL_TAG)) {
            blackhole.consume(bundle.signatures());
        }
    }

    @Benchmark
    @Measurement(batchSize = 1)
    public void inference(BertModel model, BertInput8x64 input, Blackhole blackhole) {
        Map<String,Tensor> outputs = model.inferenceFunc.call(input.feedDict);
        blackhole.consume(outputs);
        for (Map.Entry<String,Tensor> e : outputs.entrySet()) {
            e.getValue().close();
        }
    }

}
