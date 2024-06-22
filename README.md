**OVERVIEW**

1. Create your own dataset using [dataset](chargoddard/WebInstructSub-prometheus) available on Hugging face.
a. Remove instructions with less than 100 tokens in response.
b. Data deduplication by doing grouping using cosine similarity (threshold>0.95)

2. Fine-tune Llama2 using QLoRA.
Video Explaination link: [https://jmp.sh/lCzS0FL1](https://jmp.sh/lCzS0FL1)
