package fusion;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

// U-Net-like neural network (architecture and forward pass only)
public class UNet {
  Linear time_embed;
  Linear time_embed_out;

  Conv2D conv_in;

  List<Module> down1;
  List<Module> down2;

  List<Module> middle;

  List<Module> up1;
  List<Module> up2;

  Conv2D conv_out;

  public UNet() {
    time_embed = new Linear("time_embed_0");
    // There is a SiLU here
    time_embed_out = new Linear("time_embed_2");
        
    conv_in =new Conv2D("conv_in" , 1, 1);

    down1 = new ArrayList<>();
    down1.add(new ResBlock("down1", 0));
    down1.add(new ResBlock("down1", 1));
    down1.add(new Conv2D("down1_2", 2, 1));

    down2 = new ArrayList<>();
    down2.add(new ResBlock("down2", 0));
    down2.add(new ResBlock("down2", 1, false));
    down2.add(new Conv2D("down2_2", 2, 1));

    middle = new ArrayList<>();
    middle.add(new ResBlock("middle", 0));
    middle.add(new ResBlock("middle", 1));

    up1 = new ArrayList<>();
    up1.add(new TranspConv2D("up1_0", 2, 1));
    up1.add(new ResBlock("up1", 1, false));
    up1.add(new ResBlock("up1", 2, false));

    up2 = new ArrayList<>();
    up2.add(new TranspConv2D("up2_0", 2, 1));
    up2.add(new ResBlock("up2", 1, false));
    up2.add(new ResBlock("up2", 2, false));

    //A SiLU goes here
    conv_out = new Conv2D("conv_out_1", 1, 1);

  }

  public INDArray predict(INDArray x, INDArray timesteps) {
    INDArray t = this.time_embed_out.forward(Layer.SiLU(this.time_embed.forward(Utils.timestepEmbedding(timesteps))));

    INDArray h = this.conv_in.forward(x);
    List<INDArray> hs = new ArrayList<>();
    
    h = this.down1.get(0).forward(h, t);
    hs.add(h);
    h = this.down1.get(1).forward(h, t);
    hs.add(h);
    h = this.down1.get(2).forward(h, t);
    
    // Down2: 128->128->256, then downsample to 256
    h = this.down2.get(0).forward(h, t);
    hs.add(h);
    h = this.down2.get(1).forward(h, t);
    hs.add(h);
    h = this.down2.get(2).forward(h, t);
    
    // Middle: 256->256->256
    h = this.middle.get(0).forward(h, t);
    h = this.middle.get(1).forward(h, t);
    
    // Decoder - carefully match the skip connections
    // Up1: 256 + skip connections
    h = this.up1.get(0).forward(h, t);  // Upsample: 256->256
    h = Layer.concat(h, hs.remove(3));
    h = this.up1.get(1).forward(h, t);
    h = Layer.concat(h, hs.remove(2));
    h = this.up1.get(2).forward(h, t);
    
    // Up2: 128 + skip connections
    h = this.up2.get(0).forward(h, t);  // Upsample: 128->128
    h = Layer.concat(h, hs.remove(1));
    h = this.up2.get(1).forward(h, t);
    h = Layer.concat(h, hs.remove(0));
    h = this.up2.get(2).forward(h, t);

    return this.conv_out.forward(Layer.SiLU(h));
  }
}
