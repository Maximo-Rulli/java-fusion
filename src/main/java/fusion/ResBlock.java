package fusion;

public class ResBlock {
  public Layer timeMLP;
  public Layer conv1;
  public Layer conv2;
  public Layer shortcut;  // null if Identity

  public ResBlock(String blockName, int blockNumber, boolean same) {
    String generalPath = blockName + "_" + blockNumber + "_";

    // Time embedding MLP: SiLU + Linear
    timeMLP = new Layer(generalPath + "time_mlp_1");

    // Conv1: (inC → outC) with kernel 3x3
    conv1 = new Layer(generalPath + "block1_2");

    // Conv2: (outC → outC) with kernel 3x3
    conv2 = new Layer(generalPath + "block2_2");

    // Shortcut if inC != outC
    if (!same) {
        shortcut = new Layer(generalPath + "shortcut");  // 1x1 conv
    } else {
        shortcut = null; // Identity
    }
  }

  public ResBlock(String blockName, int blockNumber){
    this(blockName, blockNumber, false);
  }
}
