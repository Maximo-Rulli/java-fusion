package fusion;

import java.util.ArrayList;
import java.util.List;

// U-Net-like neural network (architecture and forward pass only)
public class UNet {

  public UNet() {
    Linear time_embed = new Linear("time_embed_0");
    // There is a SiLU here
    Linear time_embed_out = new Linear("time_embed_2");
        
    Conv2D conv_in =new Conv2D("conv_in" , 1, 1);

    List<Module> down1 = new ArrayList<>();
    down1.add(new ResBlock("down1", 0));
    down1.add(new ResBlock("down1", 1));
    down1.add(new Conv2D("down1_2", 2, 1));

    List<Module> down2 = new ArrayList<>();
    down2.add(new ResBlock("down2", 0));
    down2.add(new ResBlock("down2", 1, false));
    down2.add(new Conv2D("down2_2", 2, 1));

    List<Module> middle = new ArrayList<>();
    middle.add(new ResBlock("middle", 0));
    middle.add(new ResBlock("middle", 1));

    List<Module> up1 = new ArrayList<>();
    up1.add(new TranspConv2D("up1_0", 2, 1));
    up1.add(new ResBlock("up1", 0, false));
    up1.add(new ResBlock("up1", 1, false));

    List<Module> up2 = new ArrayList<>();
    up2.add(new TranspConv2D("up2_0", 2, 1));
    up2.add(new ResBlock("up2", 0, false));
    up2.add(new ResBlock("up2", 1, false));

    //A SiLU goes here
    Conv2D conv_out = new Conv2D("conv_out_2", 1, 1);

  }

  public double[] predict(double[] x, int t) {
    return x;
  }

  private double[] convBlock(double[] x, int inChannels, int outChannels) {
    return x;
  }

  private double[] upSample(double[] x, int inChannels, int outChannels) {
    return x;
  }

  private double[] concat(double[] x1, double[] x2) {
    return x1;
  }
}
