The "Speed Demon" Rulebook
Vectorize everything: If you find yourself writing a for loop, you've already lost the speed war. use tensor.

No Prints: Printing to the console from inside the loop requires a CPU sync. It will kill your performance.

Logging: Use TensorBoard/WandB sparsely. Log every 100 or 1,000 steps, not every step.
Minimized Function Overhead: By calculating the reward and gathering observations inside step(), you avoid the Python overhead of multiple function calls per environment step.

No Data Transfer: You correctly noted: "No .cpu() or .numpy() here." Keeping everything in torch tensors on the GPU allows Genesis to pipe physics data directly into your reward logic without the CPU ever touching the coordinates.