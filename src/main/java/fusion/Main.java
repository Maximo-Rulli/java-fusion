package fusion;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
  public static void main(String[] args) throws IOException, InterruptedException {
    final int NUM_SAMPLES = 5;
    UNet Model = new UNet();
    Correctness CorrectnessTests = new Correctness(Model);
    CorrectnessTests.run();
    Tests.run();

    //DDPM JavaFusion = new DDPM(Model, 1500);

    for (int i = 0; i < NUM_SAMPLES; i++) {
        int sampleIndex = i;  // capture loop variable for use in lambda
        Runnable task = () -> {
            // Create a separate DDPM instance per thread if not thread-safe
            DDPM JavaFusion = new DDPM(Model, 1500);
            JavaFusion.sample(100, sampleIndex); // assuming 'run' method does inference
        };

        Thread thread = new Thread(task);
        thread.start();
    }
    
    //ImageSaver.saveImage(JavaFusion.sample(100), "mnist.png");
  }
}
