---
layout: page
title: Druga laboratorijska vježba
description: Upute za drugu laboratorijsku vježbu.
nav_exclude: true
---

# Druga laboratorijska vježba: detekcija kutova i rubova u slici

Jedan od osnovnih postupaka računalnog vida
je detekcija značajki - odnosno istaknutih dijelova slike.
Detekcija značajki osnova je za rješavanje mnogih složenijih problema računalnog vida poput: praćenja, slikovnog pretraživanja, uparivanja slika i dr.
Istaknuti dijelovi slika često odgovaraju kutovima i rubovima objekata u sceni. 
Stoga ćemo u ovoj vježbi razmatrati dva algoritma: Harrisov algoritam za detekciju kutova ([članak](http://www.bmva.org/bmvc/1988/avc-88-023.pdf), [wiki](https://en.wikipedia.org/wiki/Harris_corner_detector), [opencv](https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html)) i Cannyev algoritam za detekciju rubova ([članak](https://ieeexplore.ieee.org/abstract/document/4767851), [wiki](https://en.wikipedia.org/wiki/Canny_edge_detector), [opencv](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)).
Implementacija ovih algoritama dostupna je u gotovo svim programskim bibliotekama za računalni vid, a kako bismo ih bolje razumijeli u okviru ove vježbe razvit ćemo vlastite implementacije. 

Ove upute fokusirane su na programsku implementaciju u Pythonu, a za pripremu i teorijsko razumijevanje algoritama možete se koristiti prezentacijom s predavanja ili gore navedenim poveznicama. Prilikom implementacije nećemo se koristiti bibliotekom opencv, a od pomoći će nam biti biblioteke:
[PIL](https://pillow.readthedocs.io/en/stable/), [scipy](https://scipy.org/), [matplotlib](https://matplotlib.org/) te [numpy](https://numpy.org/).

Prilikom razvoja, naše algoritme testirat ćemo na jednostavnijoj slici koja prikazuje logo FER-a, te u svijetu računalnog vida popularnoj slici kuće.

<img src="../../assets/images/lab2/fer_logo.jpg" alt="house" width="400"/>
<img src="../../assets/images/lab2/house.jpg" alt="house" width="300"/>

Preuzmite obje slike i pohranite ih lokalno na Vaše računalo.
Nakon što ste implementirali oba algoritma, demonstrirajte njihov rad na obje zadane slike.

## Harrisov detektor kutova

Implementaciji Harrisovog algoritma za detekciju kutova pristupit ćemo kroz niz koraka.

### 1. Učitavanje slike

Sliku u obliku višedimenzionalnog [numpy polja](https://numpy.org/doc/stable/reference/arrays.html) možemo učitati na sljedeći način:
```
from PIL import Image
import numpy as np
img = np.array(Image.open("path/do/slike"))
```
Zadaci:
- Provjerite koje su dimenzije tenzora `img`. Koja je visina, a koja širina slike?
- Ako je slika u formatu RGB, pretvorite je u sivu sliku uprosječivanjem po kanalima.
- Koja je minimalna, a koja maksimalna vrijednost intenziteta u slici?
- Ispišite na ekran intenzitete gornjeg lijevog isječka iz slike veličine $$10\times10$$ piksela. Umjesto for petlje, koristite izrezivanje (*eng. slicing*).
- Izvršite naredbu `print(img.dtype)`. Koji je tip podataka u polju `img`? Kako bismo izbjegli preljev u budućim operacijama, prebacite `img` u `float` (upute: koristite [ndarray.astype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html)). Još jednom provjerite koji je tip podataka u polju `img`.

### 2. Gaussovo zaglađivanje

Gaussovo zaglađivanje slike provest ćemo funkcijom [gaussian_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html) iz `scipy.ndimage`. 

Zadaci:
- Prikažite rezultate gaussovog zaglađivanja za različite vrijednosti argumenta `sigma`. Kako vrijednost toga argumenta utječe na rezultat? Za prikaz slike možete koristiti funkciju `imshow` iz `matplotlib.pyplot`.

Primjer gaussovog zaglađivanja za `sigma=5`:

<img src="../../assets/images/lab2/fer_logo_sigma5.jpg" alt="FER logo zaglađane sa sigma=5." width="400"/>

### 3. Izračun gradijenata

Za izračun gradijenata koristit ćemo se funkcijom [ndimage.convolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html).
Vrijednosti konvolucijskih jezgri za izračun gradijenata $$\mathrm{I}_x$$ i $$\mathrm{I}_y$$ postavite prema [Sobelovom operatoru](https://en.wikipedia.org/wiki/Sobel_operator).

Zadaci:
- Prikažite gradijente slike $$\mathrm{I}_x$$ i $$\mathrm{I}_y$$.
- Izračunajte druge momente gradijenata: $$\mathrm{I}_x^2$$, $$\mathrm{I}_x \mathrm{I}_y$$ te $$\mathrm{I}_y^2$$.

Primjer gradijenata po osima $$\mathrm{I}_x$$ i $$\mathrm{I}_y$$:

<img src="../../assets/images/lab2/fer_grad_x.jpg" alt="FER logo gradijent po x-u." width="300"/>
<img src="../../assets/images/lab2/fer_grad_y.jpg" alt="FER logo gradijent po y-u." width="300"/>


### 4. Sumiranje gradijenata u lokalnom susjedstvu

Prije izračuna Harrisovih odziva potrebno je u svakom pikselu $$\mathbf{q}$$ izračunati karakterističnu matricu $$\mathbf{G}$$:

$$
  \begin{equation}
  \mathbf{G}(\mathbf{q})=
    \left [\begin{array}{cc}
     \sum_W \mathrm{I}_x^2   & 
     \sum_W \mathrm{I}_x \mathrm{I}_y\\ 
     \sum_W \mathrm{I}_x \mathrm{I}_y & 
     \sum_W \mathrm{I}_y^2 
    \end{array} \right]=
    \left [\begin{array}{cc}
     a&c\\ c&b
    \end{array} \right]
  \end{equation}
$$

Primijetite da elementi matrice ne odgovaraju vrijednosti momenta u tome pikselu, nego sumi momenata u njegovom lokalnom susjedstvu. 
Veličina toga susjedstva je jedan od parametara algoritma.
Jedan način za izračunati te sume je koristeći for petlje, *slicing* i redukciju sumom. 
Međutim, primijetite da to zapravo odgovara konvoluciji s odgovarajućom jezgrom.

Zadaci:
- Inicijalizirajte odgovarajuću konvolucijsku jezgru i pozivom `ndimage.convolve` izračunajte druge momente gradijenata sumirane po lokalnom susjedstvu.

### 5. Izračun Harrisovog odziva

Harrisov odziv u svakom pikselu odgovara razlici između determinante matrice i kvadriranog traga matrice pomnoženog konstantom $$k$$ koja je parametar algoritma. Odnosno u skladu s prethodnom definicijom matrice: 

$$r(\mathbf{q})=a b -c^2 -k(a+b)^2$$

Zadaci:
- Koristeći rezultate prethodnog koraka izračunajte Harrisov odziv u svakom pikselu.
- Slikovno prikažite Harrisove odzive.

Primjer Harrisovih odziva:

<img src="../../assets/images/lab2/fer_logo_harris_odziv.jpg" alt="FER logo harrisov odziv." width="400"/>


### 6. Potiskivanje nemaksimalnih odziva

U ovome koraku cilj je potisnuti odzive koji bi mogli uzrokovati lažno pozitivne detekcije.
To ćemo napraviti u dva koraka. 

Prvo ćemo sve odzive koji su manji od zadanog praga (eng. threshold) postaviti na nulu.
Ovaj parametar algoritma dosta ovisi o slici, pa ga je potrebno prilagoditi svakoj slici. 

U drugom koraku ćemo potisnuti sve odzive koji nemaju maksimalnu vrijednost u svome lokalnom susjedstvu.
Veličina lokalnog susjedstva je još jedan od parametara algoritma. 
Slično kao u konvoluciji, ovo možemo implementirati pomicanjem prozora centriranog u odredišnom pikselu,
te postavljanjem toga piksela na nulu, ako je njegova vrijednost manja od maksimalne vrijednosti unutar prozora.

Zadaci:
- Implementirajte potiskivanje odziva manjih od praga bez upotrebe for petlje.
- Implementirajte potiskivanje nemaksimalnih odziva unutar lokalnog susjedstva uz upotrebu maksimalno dvije ugnježđene for petlje.

### 7. Selektiranje k-najvećih odziva

Posljednji korak algoritma je selekcija k-najvećih odziva.

Zadaci:
- Dohvatite sve koordinate odziva različitih od nule koristeći funkciju [numpy.nonzero](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html).
- Izdvojite koordinate piksela koji odgovaraju k-najvećih odziva.
- Vizualizirajte rezultate algoritma iscrtavajući krugove u koordinatama detektiranih kutova povrh zadane slike.

Primjer detektiranih Harrisovih kutova:

<img src="../../assets/images/lab2/fer_logo_harris_corners.jpg" alt="FER logo Harrisovi kutovi." width="500"/>

Parametri algoritma: sigma=1, prag=1e10, k=0.04, topk=100, veličina_prozora_za_sumiranje=(5, 5), veličina_prozora_za_potiskivanje_nemaksimalnih_odziva=(14, 14)


<img src="../../assets/images/lab2/house_harris_corners.jpg" alt="House Harrisovi kutovi." width="400"/>

Parametri algoritma: sigma=1.5, prag=1e9, k=0.04, topk=100, veličina_prozora_za_sumiranje=(5, 5), veličina_prozora_za_potiskivanje_nemaksimalnih_odziva=(32, 32)

## Cannyev detektor rubova

Implementaciji Cannyevog algoritma za detekciju rubova također ćemo pristupiti kroz niz koraka.
Prva tri koraka algoritma implementirajte jednako kao kod Harrisovog algoritma.

### 1. Učitavanje slike
### 2. Gaussovo zaglađivanje
### 3. Izračun gradijenata

### 4. Izračun iznosa i smjera gradijenta
U ovome koraku ćemo u svakome pikselu izračunati iznos gradijenta $$|G| = \sqrt{I_x^2 + I_y^2}$$ te smjer gradijenta $$\theta = \arctan(I_y/I_x)$$.

Zadaci:
- Izračunajte iznos i smjer gradijenta u svakom pikselu prema zadanoj formuli.
- Normalizirajte polje iznosa gradijenta na interval $$[0-255]$$. To možete napraviti na način da ga podijelite s maksimalnom vrijednosti prisutnom u polju i pomnožite s 255.
- Vizualizirajte normalizirane iznose gradijenta.


Primjer vizualizacije iznosa gradijenta:

<img src="../../assets/images/lab2/house_magnitudes.jpg" alt="House Canny gradient magnitudes." width="400"/>


### 5. Potiskivanje nemaksimalnih odziva

Cilj ovoga koraka je potisnuti odzive u neželjenim pikselima koji ne čine nijedan od rubova u slici.
Cannyev algoritam potiskuje odzive koji nisu lokalni maksimum u susjedstvu koje se proteže po pravcu gradijenta.
Postupak provodimo na sljedeći način. U svakome pikselu očitamo iznos kuta gradijenta. 
Prema tome kutu određujemo jedan od mogućih diskretnih pravaca gradijenata.
Susjedstvo onda čine dva nasuprotna susjedna piksela koji leže na tome pravcu.
Primjerice, prema slici ispod, 
ako za kut $$\theta$$ vrijedi $$22.5^{\circ} < \theta < 67.5^{\circ}$$ ili $$-157.5^{\circ} < \theta < -112.5^{\circ}$$,
tada susjedstvo čine pikseli gore-desno i dolje-lijevo. 
Pri programskoj implementaciji vrijedi obratiti pažnju na 
područje vrijednosti koje vraća funkcija inverznog tangensa,
te na činjenicu da indeksi redaka u polju rastu od gore prema dolje.

<img src="../../assets/images/lab2/canny_angles.jpg" alt="House Canny gradient magnitudes." width="400"/>

Konačno, razmatrani piksel će "preživjeti" samo ako je njegov smjer veći od oba susjedna piksela.
Inače, njegov iznos se postavlja na nulu.

Ovakav postupak potiskivanja nemaksimalnih odziva za posljedicu ima tkz. stanjivanje rubova.

Zadaci:
- Implementirajte opisani postupak potiskivanja nemaksimalnih odziva.
- Vizualizirajte iznose gradijenata nakon postupka potiskivanja nemaksimalnih odziva.


Primjer vizualizacije iznosa gradijenta nakon potiskivanja nemaksimalnih odziva:

<img src="../../assets/images/lab2/house_magnitudes_nms.jpg" alt="House Canny gradient magnitudes after NMS." width="400"/>

### 6. Uspoređivanje s dva praga - histereza

U posljednjem koraku algoritam donosi konačnu odluku koji od piksela zaista jesu rubovi, a koji ne.
To činimo uspoređivanjem s dva praga, odnosno histerezom. 
Razlikujemo gornji prag kojeg ćemo označiti s max_val, te donji prag kojeg ćemo označiti s min_val.

Svi pikseli čiji je iznos veći od gornjeg praga čine tkz. jake rubove. 

Sve piksele čija je iznos manji od donjeg praga odbacujemo, te smatramo da nisu rubovi.

Piksele čiji se iznosi gradijenta nalaze između dva praga smatramo slabim rubovima.
Odluku za njih donosimo na temelju njihove povezanosti. 
Ako se nalaze u susjedstvu nekog od jakih rubova, onda oni također čine rub.
Ako to nije slučaj, odbacujemo ih i smatramo da nisu rubovi.

Zadaci:
- Implementirajte opisani postupak uspoređivanja s pragom.
- Vizualizirajte samo jake rubove.
- Vizualizirajte konačan rezultat algoritma.

Primjer samo jakih rubova:

<img src="../../assets/images/lab2/house_strong_edges.jpg" alt="House Canny strong edges." width="400"/>

Primjer rezultata detekcije rubova Cannyevim algoritmom:

<img src="../../assets/images/lab2/house_edges.jpg" alt="House Canny edges." width="400"/>

Ovi rezultati postignuti su sa sljedećim vrijednostima parametara algoritma: sigma=1.5, min_val=10, max_val=90
