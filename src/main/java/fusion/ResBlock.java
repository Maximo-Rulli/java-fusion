package fusion;

import org.nd4j.linalg.api.ndarray.INDArray;

public class ResBlock extends Module{
  public Linear timeMLP;
  public Conv2D conv1;
  public Conv2D conv2;
  public Conv2D shortcut;  // null if Identity

  public ResBlock(String blockName, int blockNumber){
    this(blockName, blockNumber, true);
  }

  public ResBlock(String blockName, int blockNumber, boolean same) {
    String generalPath = blockName + "_" + blockNumber + "_";

    // Time embedding MLP: SiLU + Linear
    timeMLP = new Linear(generalPath + "time_mlp_1");

    // Conv1: (inC → outC) with kernel 3x3
    conv1 = new Conv2D(generalPath + "block1_2", 1, 1);

    // Conv2: (outC → outC) with kernel 3x3
    conv2 = new Conv2D(generalPath + "block2_2", 1, 1);

    // Shortcut if inC != outC
    if (!same) {
        shortcut = new Conv2D(generalPath + "shortcut");  // 1x1 conv
    } else {
        shortcut = null; // Identity
    }
  }

  public INDArray forward (INDArray x, INDArray t){
    INDArray h = conv1.forward(Layer.SiLU(x));                      // [1, C, H, W]
    
    INDArray timeEmbedding = timeMLP.forward(Layer.SiLU(t));         // [1, C]
    timeEmbedding = timeEmbedding.reshape(timeEmbedding.size(0), timeEmbedding.size(1), 1, 1);
    
    h = h.add(timeEmbedding);                            // Broadcasting addition
    
    h = conv2.forward(Layer.SiLU(h));                               // [1, C, H, W]

    if (shortcut != null){
      INDArray shortcutOut = shortcut.forward(x);
      h = h.add(shortcutOut);
    }

    return h;  
  }

}
