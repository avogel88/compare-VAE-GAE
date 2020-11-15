# Vergleich zwischen generativen Modellen: Variational Autoencoder und Iterative Gaussianization Autoencoder
Comparison between Variational and Iterative Gaussianization AE

## Aufgabe / Task
Stochastische generative Modelle haben viele Verwendungen, unter anderen in Compression, Denoising,
Semi-Supervised und Unsupervised Learning, Inpainting und Texture Synthesis [TOB15].
Die genutzten Verfahren sind heterogen, aber die Grundidee der meisten Verfahren ist ähnlich. Das Ziel
ist, die Verteilung der Daten aus einer gegebenen Stichprobe zu approximieren. Dazu wird eine Transformation zwischen dieser unbekannten Verteilung und einer bekannten Verteilung (meistens eine multivariate Normalverteilung) gelernt. Die Qualitat des generativen Modells hängt nun wesentlich davon ab, dass
die Transformation die Verteilungen korrekt überführt.
Ziel dieser Arbeit ist der Vergleich von zwei generativen Modellen. Das erste ist ein Variational Autoencoder (VAE) [KW13]. Der VAE wird mit zwei Loss-Funktionen trainiert. Die erste loss-Funktion erzwingt
eine gute Rekonstruktion der Eingabe und somit eine hohe Likelihood unter der Rücktransformation. Die
zweite Loss-Funktion erzwingt, dass die resultierenden encodings in etwa normalverteilt sind. Der Encoder ist daher eine Transformation der unbekannten Verteilung zu einer Normalverteilung und der Decoder eine Transformation einer Normalverteilung zu der unbekannten Verteilung.
Die zweite Methode ist ein Autoencoder kombiniert mit Iterative Gaussianization [LCVM11] im Encoding Raum.
Die Aufgabe ist nun diese Verfahren zu implementieren und auf ausgewahlten Datensätzen zu vergleichen.
Zum einen sollte auf einem Datensatz mit bekannter Verteilung die Likelihood zwischen zufälligen Stichproben in der Originalverteilung und deren Bild nach Anwendung der Transformation zur Normalverteilung
verglichen werden. Ebenso soll die Likelihood zwischen zufälligen Stichproben in der Normalverteilung
und deren Bild nach Anwendung der Transformation zur Originalverteilung verglichen werden.
Zum Anderen soll qualitativ die Ähnlichkeit von generierten Beispielen und originalen Beispielen in
mindestens einem Bilder-Datensatz untersucht werden.

## References

| | |
| --- | --- |
| [KW13] | KINGMA, Diederik P. ; WELLING, Max: Auto-encoding variational bayes. In: arXiv preprint arXiv:1312.6114 (2013)
| [LCVM11] | LAPARRA, Valero ; CAMPS-VALLS, Gustavo ; MALO, Jesús: Iterative gaussianization: from ICA to random rotations. In: IEEE transactions on neural networks 22 (2011), Nr. 4, S. 537–549
| [TOB15] | THEIS, Lucas ; OORD, Aäron van den ; BETHGE, Matthias: A note on the evaluation of generative models. In: arXiv preprint arXiv:1511.01844 (2015)

## Example for VAE

In Bash / Powershell:
```bash
tensorboard --logdir checkpoints
```
Python:
```python
import numpy as np
from modules.experiment import Autoencoder, layers
from modules.tensorflow.keras.datasets.MNIST import import_mnist
from modules.tensorflow.keras.layers import Variational
from modules.experiment.plot import plot_strip
from numpy.random import randn
from scipy.stats import kstest

(x, _), (y, _) = import_mnist('mnist')

dim_z = 10
num_digits = 20

enc, dec = layers(Variational, dim_z)
vae = Autoencoder(
    enc, dec, dim_z,
    name='VariationalAutoencoder',
    short_name='VAE',
    loss=Variational.MSE)
vae.fit(
    x, x,
    epochs=100,
    validation_data=(y, y))

eps = randn(num_digits, dim_z).astype(np.float32)
rec = vae.predict(y)
enc = vae.encode(x)
gen = vae.decode(eps)
ks = np.apply_along_axis(kstest, 0, enc, 'norm')
plot_strip((y, rec, gen), num_digits)
print(ks)
```
