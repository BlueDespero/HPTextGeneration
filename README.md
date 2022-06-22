Model is based on Harry Potter books. 
To see generate text run the evaluate.py script.
This evaluation takes a long time because of the constraints put on the model. 
To try it without constraints  run the predict function with the 'unbounded' argument Set to True.
Places in the code which are responsible for meeting the constraints described in the task are marked with the comments.
All the requirements are met + the additional bigram constraint.  

## Sources:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
https://dzlab.github.io/dltips/en/tensorflow/create-bert-vocab/
https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
