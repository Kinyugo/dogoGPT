# dogoGPT

A tiny GPT implementation that uses the U-Net block from **Msanii** paper.

It uses the U-Net block from Msanii, but makes a few changes:

1. Causal convolutions.
2. LayerNorm instead of InstanceNorm for normalization.
3. OG attention instead of linear-attention.
4. Feed Forward instead of U-Net.

## Samples

This model has been trained on [shakespeare](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt).

**prompt**

```
"\n"
```

**temperature=1,topk=None**

```

Second Servingman:
Warwick now shame so.

CAPULET:
And too soon marr'd are those so early made.
The earth hath swallow'd all my hopes but she,
She is the lark that sings so out of tune,
Straining harsh discords and unpleasing sharps.
Some say the lark and loathed toad change eyes,
O, now I would they had changeth thus his manners.

CAMILLO:
It is fifteen years
Do consider that which may
Unfurnish me of reason. They are come.
Your mother well hath pray'd, and prove you true.

DUCHESS OF YORK:
Come, my old so
```

**temperature=0.8,top_k=None**

```

You find whose down thought of this from Henry's heart.

DORSET:
I will not rise, unless your highness grant.

KING EDWARD IV:
So other foes may set upon our backs.
Stand we in good array; for they no doubt
Will issue out again and bid us battle:
If not, the city being but of small defence,
We'll quickly rouse the traitors in the same.

WARWICK:
O passing traitor, perjured and unjust.

CORIOLANUS:
What do you prate of service?

BRUTUS:
I talk of that darkest with a goodly son,
Didst provokes that Caliban
Sh
```

**temperature=1,top_k=20**

```

Play the man that true Bear
The addition nobly every
very pitter and y'er that joy is dearest, and go to change our loves. Beseech you, sir,
Remember since you owed no more to time
Murn back, than to your face.

PARIS:
Poor soul, thy face is much abused with tears.

JULIET:
Then, window, let day in, and let life out.

ROMEO:
Farewell, farewell, and sit you fast,
For I will hence to-night.

BALTHASAR:
I would your duty were told on't, and he shall:nd with the other sends
I warrant your honour see any harm in
```

## References

```bibtex
@article{maina2023msanii,
  title={Msanii: High Fidelity Music Synthesis on a Shoestring Budget},
  author={Maina, Kinyugo},
  journal={arXiv preprint arXiv:2301.06468},
  year={2023}
}
```
