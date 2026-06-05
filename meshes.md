# Different Mesh Types Explained

## Square
Quick default run (Square domain, single wavepacket)
```bash
python3 -m scripts.main --mesh_type square 
```

![Particle in a Square Box Evolution](media/square_1pkt.gif)

## Lieb-Robinson Bounds
Compute Lieb-Robinson bounds for square domain with offset wavepacket

```bash
python3 -m scripts.run_lieb_robinson_study 
```
![Particle with LR bounds](media/LR.gif) 
![Particle without LR bounds](media/LR_woCone.gif)

## Circle

Run a circular domain configuration 
```bash
python3 -m scripts.main --mesh_type circular
```

![Particle in a Circular Box Evolution](media/circular_1pkt.gif)

## Stadium

Run a high-energy dual-wavepacket collision inside a chaotic Bunimovich Stadium
```bash
python3 -m scripts.main --mesh_type stadium --num_packets 2 --T 10.0 --frames 500 
```

![Particle in a Stadium Evolution](media/stadium_2pkt_colliding.gif)