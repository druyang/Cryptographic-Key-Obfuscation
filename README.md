# CUDA-based Cryptographic Key Obfuscation
A novel implementation of program obfuscation that leverages the unique parallel programming paradigm offered by the CUDA GPGPU computing language to generate obscure and unintelligable binary when compiled. Obfuscated binaries written in the CUDA language may be more difficult to 'crack' than ordinary CPU-based obfuscated code. This project was inspired by the concept of [Translingual Obfuscation](https://faculty.ist.psu.edu/wu//papers/to-eurosp16.pdf`).

For details, see our [Final Report](./Project-Report.pdf).

**Contributors**: Andrw Yang, Matt Kenney, and Jeff Liu

> "This project wins the best demonstration and presentation among all [graduate and undergraduate] teams. It picks a very challenging topic to showcase the power of GPU computing and in the end fully demonstrates the efficacy of GPU  in a very novel context."

-[Bo Zhu](https://www.dartmouth.edu/~boolzhu/), COSC 89.25/189.25 - GPU Programming and High-Performance Computing

**Proof of Concept**

![](https://raw.githubusercontent.com/druyang/Cryptographic-Key-Obfuscation/master/NovelTheory.png)

An end-to-end encryption and statistics data platform (for medical statistic and contact tracing) was chosen as a proof of concept application of this approach.
Our concept and improvements over the status quo is illustrated by the diagram below: 

**Results**

|   | CPU  | GPU |
|---|---|---|
| Decryption (ms) | 500~  | 2.7  |
| Statistics* (ms) | 0.13  | 0.25  |
| Total Time (ms)  | 500.13  | 2.95  |
| Security  | RSA  | RSA+Obfuscation, "end to end" paradigm described below  |

* We chose the t-test as the "statistics" analysis. From a performance perspective, the t-test favors the CPU. Linear regression would be a statistical measurement that takes advantage of the GPU's computational power. 

The proof of concept exhibited significant runtime (~170x speedup), yet more importantly significant security improvements over CPU code. 
The following are the many layers of security behind our "end to end", black box encryption/obfuscation paradigm: 

 * Encryption key computationally obfuscated inside the CUDA binary
 * Translingual obfuscation (in CUDA)
 * Sensitive data is routed through GPU memory only and never touches CPU/RAM
 * RSA Encryption

View the [Final Report](./Project-Report.pdf) for details.


