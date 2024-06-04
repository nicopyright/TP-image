<div align="justify">
Toutes les réponses, images et bouts de code requis pour le compte rendu sont dans le README. Aucun fichier du dépôt n'est à aller vérifier. 

# Prise en main "rapide"

- Tester le programme qui lit puis affiche une image couleur ainsi que ses trois composantes. Les tests seront réalisés à partir des images disponibles dans le répertoire fourni.

<p align="center" >
    <image style="width:80%;" src="img/CerisierP_couleurs.png" ></image>
  <div align="center"> 
    <i>Image d'origine et ses trois composantes du fichier "CerisierP.jpg"</i>
  </div>
</p>
  
- Tester la fonction histogramme des niveaux de gris d’une image.

Notre fonction histogramme:

```python
def histogramLvlOfGrey(img):
    imgG = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    hist = np.zeros(256)
    for line in imgG:
        for pixel in line:
            hist[pixel] += 1
    plt.figure()
    plt.bar(np.arange(256), hist)
    plt.show()
    return hist
```

Et Voici un histogramme correspondant à l'image "cerisierp.jpg" :

<p align="center" style="display:inline-block;">
    <image style="width:80%;" src="img/histo_CerisierP.png" ></image>
  <div align="center" > 
    <i>Histogramme de l'image "cerisierp.jpg"</i>
  </div>
</p>  
- Écrire et tester un programme permettant de binariser une image. Tout d’abord le seuil sera entré en paramètre (choisi à partir de l’examen visuel de l’histogramme de l’image), puis il sera obtenu automatiquement à partir d’une fonction basée sur la méthode des moments statistiques (cf cours).

Voilà notre code pour la binairisation d'une image avec le calcul de seuil automatique :

```python
def statisticImageMoments(img,hist,i):
    sum = 0
    for j in range(0,256):
        sum += hist[j]*(j)**i
    return 1/(img.shape[0]*img.shape[1]) * sum

def binaryImg(img, seuil):
    imgG = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    imgB = np.zeros_like(imgG)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if imgG[i,j] > seuil:
                imgB[i,j] = 255
    return imgB

def autoBinaryImg(img):
    hist = histogramLvlOfGrey(img)
    sim = (statisticImageMoments(img, hist, 1))
    return binaryImg(img, sim)
```

Et pour nos tests, notre main : 

```python
if __name__ == "__main__":
    filename = "4.jpg"
    name = filename.split(".")[0]
    img = cv.imread("imagesTP/"+filename)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # Convert to RGB
    plt.imshow(img)
    bI = autoBinaryImg(img)
    plt.imshow(bI, cmap="gray")
    plt.imsave(fname="imagesTP/output/" + name + "_binary.png", arr=bI, format="png", cmap="gray")
    plt.show()
```

Et voilà notre résultat sur l'image :
<p align="center" style="display:inline-block;">
    <image style="width:80%;" src="img/result_binarisation.png" ></image>
  <div align="center"> 
    <i>Image binarisée avec seuil calculé de l'image "CerisierP.jpg"</i>
  </div>
</p>

- Écrire un programme qui réalise les opérations suivantes :
  
  - Calcul et affichage de l’histogramme d’une image (appel de la fonction   histogramme réalisée en question 3)
  - Égalisation d’histogramme sur cette image
  - Affichage de la fonction de répartition, de l’histogramme de l’image égalisée.
  
  #Non traité#

# II. Transformée de Fourier

### a. Harmoniques pures


<p align="center" style="display:inline-block;">
    <image style="width:42%;" src="img/image1.png" ></image> <image style="width:40%;" src="img/spectre1.png"></image>
  <div align="center" > 
    <i>Image et spectre 2D de l'image générée avec les paramètres f1=0.1 et f2=0</i>
  </div>
</p>


<p align="center">
  <image style="width:42%;" src="img/image2.png"></image> <image style="width:40%;" src="img/spectre2.png"></image>
  <div align="center" > 
  <i>Image et spectre 2D de l'image générée avec les paramètres f1=0 et f2=0.1</i>
  </div>
</p>

<p align="center">
  <image style="width:42%;" src="img/image3.png"></image> <image style="width:40%;" src="img/spectre3.png"></image>
  <div align="center" > 
  <i>Image et spectre 2D de l'image générée avec les paramètres f1=0.3 et f2=0.3</i>
  </div>
</p>

<p align="center">
<image style="width:42%;" src="img/image4.png"></image> <image style="width:40%;" src="img/spectre4.png"></image>
  <div align="center" > 
  <i>Image et spectre 2D de l'image générée avec les paramètres f1=-0.3 et f2=0.1</i>
  </div>
</p>


### b. Contour

#### Contour Vertical

```python
    image = np.zeros([128,128])
    [height, width] =image.shape
    for i in range(height):
        for j in range(width):
            if j < width/2 :
                image[i,j] = 255
```

<p align="center">
  <image style="width:42%;" src="img/contour_vertical.png"></image> <image style="width:40%;" src="img/spectre_contour_vertical.png"></image>
  <div align="center" > 
    <i>Image et spectre 2D de l'image "contour vertical"</i>
  </div>
</p>

On observe un contour vertical dans l'image, ce qui se traduit par une ligne horizontale dans son spectre. La ligne est en fait en pointillée étant donné la discretisation de l'image.

#### Contour horizontal

```python
    image = np.zeros([128,128])
    [height, width] =image.shape
    for i in range(height):
        for j in range(width):
            if i < height/2 :
                image[i,j] = 255
```

<p align="center">
  <image style="width:42%;" src="img/contour_horizontal.png"></image> <image style="width:40%;" src="img/spectre_contour_horizontal.png"></image>
  <div align="center" > 
    <i>Image et spectre 2D de l'image "contour horizontal"</i>
  </div>
</p>

On observe un contour horizontal dans l'image, ce qui se traduit par une ligne verticale dans son spectre. Comme pour le spectre précédent, la ligne est en pointillée étant donné la discretisation de l'image.

#### Contour oblique

```python
    image = np.zeros([128,128])
    [height, width] =image.shape
    for i in range(height):
        for j in range(width):
            if j+i < (height + width)/2 :
                image[i,j] = 255
```
<p align="center">
  <image style="width:42%;" src="img/contour_oblique.png"></image> <image style="width:40%;" src="img/spectre_contour_oblique.png"></image>
  <div align="center" > 
    <i>Image et spectre 2D de l'image "contour oblique"</i>
  </div>
</p>
On observe un contour oblique dans l'image, ce qui se traduit par une ligne horizontale, une ligne verticale et une ligne oblique opposée à celle de l'image dans son spectre. Théoriquement, il n'y aurais qu'une ligne oblique mais étant donné que l'image est discrète et composée de pixels, la ligne "oblique" visible est en fait un escalier composé de lignes horizontales et verticales.

### c. Texture

Toutes les images de cette partie ont étés mises en nuances de gris pour des raisons de clareté et de code.

#### Metal0007GP
<p align="center">
  <image style="width:42%;" src="img/imageMetal.png"></image> <image style="width:40%;" src="img/imageMetal_spectre2D.png"></image>
  <div align="center" > 
    <i>Image et spectre 2D de l'image "Metal0007GP"</i>
  </div>
</p>

On peut observer des lignes horizontales et verticales dans l'image, ce qui se traduit dans le spectre par des lignes verticales et horizontales. On peut même distinguer l'inclinaison des lignes de l'image depuis les lignes du spectre. 

#### Water0000GP
<p align="center">
  <image style="width:42%;" src="img/imageWater.png"></image> <image style="width:40%;" src="img/imageWater_spectre2D.png"></image> 
  <div align="center" > 
    <i>Image et spectre 2D de l'image "Water0000GP"</i>
  </div>
</p>

Contrairement à l'image précédente, cette image est plus diffuse. On peut néanmoins remarquer que les vagues sont des formes horizontales. Ces formes se traduisent dans le spectre par une forme de sablier vertical

#### Leaves0012GP
<p align="center">
  <image style="width:42%;" src="img/imageLeaves.png"></image> <image style="width:40%;" src="img/imageLeaves_spectre2D.png"></image>
  <div align="center" > 
    <i>Image et spectre 2D de l'image "Leaves0012GP"</i>
  </div>
</p>

\\\Ajouter description

## 2. Phénomène de repliement

#### Avec image en 128x128 et Fe=1

<p align="center">
  <image style="width:43%;" src="img/imageRepliement128.png"></image> <image style="width:40%;" src="img/imageRepliement128_spectre2D.png"></image>
  <div align="center" > 
    <i>Image et spectre 2D de l'image générée avec <code>atom(128,128,0.15,0.37)</code></i>
  </div>
</p>

#### Avec image en 64x64 et Fe=0.5

<p align="center">
  <image style="width:43%;" src="img/imageRepliement64.png"></image> <image style="width:40%;" src="img/imageRepliement64_spectre2D.png"></image>
  <div align="center" > 
    <i>Image et spectre 2D de l'image générée avec <code >atom(64,64,0.15,0.37)</code ></i>
  </div>
</p>

La première image et son spectre (celle en 128x128) montrent une image d'un signal "continu", contrairement à la deuxième image et spectre (en 64x64) qui, comme attendu représente le même spectre mais avec une résolution amoidrie. On en conclue alors que la résolution des images ne change pas directement leurs spectres, mais change la précision de celui-ci.

# III. Changement d’espaces colorimétriques (comparaison HSV/IHLS)

## Code final

```python
if __name__ == "__main__":
    plt.close('all')
    img = cv.imread('imagesTP/confiserie-smarties-lentilles_121-50838.jpg')
    
    # Convertir l'image de BGR à RGB pour l'affichage
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Afficher l'image originale en RGB
    plt.figure()
    plt.imshow(rgb)
    plt.title("Image originale")
    plt.colorbar()
    
    # Convertir l'image de BGR à HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Diviser l'image HSV en composants H (teinte), S (saturation) et V (valeur)
    h, s, v = cv.split(hsv)
    
    # Afficher le composant teinte avec une cmap hsv
    plt.figure()
    plt.imshow(h, cmap='hsv')
    plt.title('Teinte')
    plt.colorbar()
    
    # Afficher le composant saturation en niveaux de gris
    plt.figure()
    plt.imshow(s, cmap='gray')
    plt.title('Saturation')
    plt.colorbar()
    
    # Afficher le composant valeur en niveaux de gris
    plt.figure()
    plt.imshow(v, cmap='gray')
    plt.title('Valeur')
    plt.colorbar()
    
    # Normaliser les canaux r, g et b
    r, g, b = cv.split(img)
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    
    # Appliquer le cours pour calculer L, S et H
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    S = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    numerator = (r - g / 2 - b / 2)
    denominator = np.sqrt((r ** 2) + (g ** 2) + (b ** 2) - (r * g) - (r * b) - (g * b))
    H_ = np.degrees(np.arccos(numerator / (denominator + 10 ** (-10))))
    H = np.where(b > g, 360 - H_, H_)
    
    # Fusionner les composants IHSL calculés
    ihsl = cv.merge([H, S, L])
    
    # Afficher l'image IHSL fusionnée
    plt.figure()
    plt.imshow(ihsl)
    plt.title('Image IHSL')
    plt.colorbar()

    plt.show()
```

Le code ci-dessus nous affiche les images demandés du sujet, à savoir : 
- l'image originale
- composante de teintes de l'image
- composante de saturation de l'image
- composante de valeur de l'image
- l'image IHSL finale

Ce qui nous donne les images suivantes :

<p align="center">
  <image style="width:40%;" src="img/smarties_original.png"></image> <image style="width:40%;" src="img/smarties_teinte.png"></image> <image style="width:40%;" src="img/smarties_saturation.png"></image> <image style="width:40%;" src="img/smarties_valeur.png"></image> <image style="width:40%;" src="img/smarties_IHSL.png"></image>
  <div align="center" > 
    <i>Images et composantes pour arriver à l'image IHSL</i>
  </div>
</p>

Quelle sont les limites de l’espace HSV ?



# IV. Segmentation d’images

## Code général

```python
import numpy as np

from matplotlib import pyplot as plt
import cv2 as cv

def histogramLvlOfGrey(img):
    hist = np.zeros(256)
    for line in img:
        for pixel in line:
            hist[pixel] += 1
    return hist

def statisticImageMoments(img,hist,i):
    sum = 0
    for j in range(0,256):
        sum += hist[j]*(j)**i
    return 1/(img.shape[0]*img.shape[1]) * sum

def binaryImg(img, seuil):

    imgB = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > seuil:
                imgB[i,j] = 255
    return imgB
def autoBinaryImg(img):
    hist = histogramLvlOfGrey(img)
    sim = (statisticImageMoments(img, hist, 1))
    return binaryImg(img, sim)

def histogramRepartition(img):
    hist = histogramLvlOfGrey(img)
    Ymax = img.shape[0]
    Xmax = img.shape[1]
    N = Ymax*Xmax
    repartition = np.zeros_like(hist)
    kmax = np.where(hist == max(hist))[0][0]

    for i in range(1, kmax):
        repartition[i] = repartition[i-1] + hist[i]
    for i in range(0, kmax):
        repartition[i] = (kmax-1)*repartition[i]/N
    plt.bar(np.arange(256), repartition)
    newimg = np.zeros_like(img)
    for i in range(Ymax):
        for j in range(Xmax):
            img[i,j] = repartition[img[i,j]]
            gris = img[i,j]
            newGris = repartition[gris]
            newimg[i,j] = newGris
    return newimg

if __name__ == "__main__":
    plt.close("all")
    filename = "confiserie-smarties-lentilles_121-50838.jpg"
    name = filename.split(".")[0]
    img = cv.imread("imagesTP/"+filename)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # Convert to RGB
    r,g,b = cv.split(img)
    h,s,v = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.figure()
    plt.imshow(g, cmap="gray")
    plt.title("Green")
    plt.figure()
    plt.imshow(h, cmap="gray")
    plt.title("Hue")

    G_hist = histogramLvlOfGrey(g)
    plt.figure()
    plt.bar(np.arange(256), G_hist)
    plt.title("Histogramme du canal G")

    G_bin = binaryImg(g, 230)
    plt.figure()
    plt.imshow(G_bin, cmap="gray")
    plt.title("Binarisation du canal G")

    H_hist = histogramLvlOfGrey(h)
    plt.figure()
    plt.bar(np.arange(256), H_hist)
    plt.title("Histogramme du canal H")

    H_bin = binaryImg(h, 50)
    plt.figure()
    plt.imshow(H_bin, cmap="gray")
    plt.title("Binarisation du canal H")

    plt.show()
```

### Segmentation des smarties jaunes

Pour segmenter les smarties jaunes, on doit trouver un canal ou il est facile de discerner la couleur voulue et les autres. Pour trouver la couleur jaune, on a vérifié à la main chaque canaux (R,G,B,H,S,V). Le canal de vert (G) semblait être le plus approprié. Une manière de vérifier cette technique est de regarder les histogrammes et vérifier s'il y a un pic à une certaine valeur de gris. Voici les résultats obtenus :

<p align="center">
  <image style="width:40%;" src="img/smarties_original.png"></image> 
  <image style="width:40%;" src="img/smarties_green.png"></image> <image style="width:40%;" src="img/smarties_histo_green.png"></image> <image style="width:40%;" src="img/smarties_bin_green.png"></image>
  <div align="center" > 
    <i>Image originale, canal G, histogramme et image binarisée avec un seuil de 230</i>
  </div>
</p>

En regardant l'histogramme, on peut observer un pic de valeur vers 255. On en déduis donc que nos smarties jaunes correspondent à une valeur élevée. Après quelques tests, on a défini un seuil à 230, ce qui nous à donné cette image binarisée.

### Segmentation des smarties bleus

La segmentation s'est faite comme pour la segmentation des smarties jaunes, à savoir regarder les différents canaux et en déduire un qui facilite la différentiation de la couleur bleue. On a alors choisis le canal Hue (H) et avons, comme pour la partie précédente, vérifié avec l'histogramme. Voici nos résultats :

<p align="center">
  <image style="width:40%;" src="img/smarties_original.png"></image> <image style="width:40%;" src="img/smarties_hue.png"></image><image style="width:40%;" src="img/smarties_histo_hue.png"></image>  <image style="width:40%;" src="img/smarties_bin_hue.png"></image>
  <div align="center" > 
    <i>Image originale, canal H, histogramme et image binarisée avec un seuil de 50</i>
  </div>
</p>

Contrairement à la partie précédente, les smarties que l'on veut retrouver sont foncés sur le canal, ce qui veut dire qu'ils ont une valeur faible. Sur l'histogramme, on recherche alors un pic bas et on trouve un pic en dessous de 50, qu'on a alors choisis comme notre seuil. Ce seuil appliqué à la binarisation nous donne alors le résultat ci-dessus.

## Amélioration de la segmentation

Pour améliorer la segmentation, on peut utiliser la dilatation ou l'érosion. Après application, voici nos résultats :

```python
from scipy import signal
import numpy as np

from matplotlib import pyplot as plt
import cv2 as cv

def histogramLvlOfGrey(img):
    hist = np.zeros(256)
    for line in img:
        for pixel in line:
            hist[pixel] += 1
    return hist

def statisticImageMoments(img,hist,i):
    sum = 0
    for j in range(0,256):
        sum += hist[j]*(j)**i
    return 1/(img.shape[0]*img.shape[1]) * sum

def binaryImg(img, seuil):

    imgB = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > seuil:
                imgB[i,j] = 255
    return imgB
def autoBinaryImg(img):
    hist = histogramLvlOfGrey(img)
    sim = (statisticImageMoments(img, hist, 1))
    return binaryImg(img, sim)

def histogramRepartition(img):
    hist = histogramLvlOfGrey(img)
    Ymax = img.shape[0]
    Xmax = img.shape[1]
    N = Ymax*Xmax
    repartition = np.zeros_like(hist)
    kmax = np.where(hist == max(hist))[0][0]

    for i in range(1, kmax):
        repartition[i] = repartition[i-1] + hist[i]
    for i in range(0, kmax):
        repartition[i] = (kmax-1)*repartition[i]/N
    plt.bar(np.arange(256), repartition)
    newimg = np.zeros_like(img)
    for i in range(Ymax):
        for j in range(Xmax):
            img[i,j] = repartition[img[i,j]]
            gris = img[i,j]
            newGris = repartition[gris]
            newimg[i,j] = newGris
    return newimg

def maskimageFrombinary(img,mask,value):
    newimg = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i,j] == value:
                newimg[i,j] = img[i,j]
    return newimg

if __name__ == "__main__":
    plt.close("all")
    filename = "confiserie-smarties-lentilles_121-50838.jpg"
    name = filename.split(".")[0]
    img = cv.imread("imagesTP/"+filename)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # Convert to RGB
    r,g,b = cv.split(img)
    h,s,v = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title("Original")

    plt.figure()
    plt.imshow(g, cmap="gray")
    plt.title("Green")
    plt.figure()
    plt.imshow(h, cmap="gray")
    plt.title("Hue")
    plt.figure()

    G_bin_erode = cv.erode(G_bin, np.ones((3,3), np.uint8), iterations=1)
    plt.figure()
    plt.imshow(G_bin_erode, cmap="gray")
    plt.title("Binarisation érodé du canal G")

    H_bin_dilate = cv.dilate(H_bin, np.ones((3,3), np.uint8), iterations=1)
    plt.figure()
    plt.imshow(H_bin_dilate, cmap="gray")
    plt.title("Binarisation dilaté du canal G")

    yellowsmarties = maskimageFrombinary(img, G_bin_erode, 255)
    plt.figure()
    plt.imshow(yellowsmarties, cmap="gray")
    plt.title("Smarties jaunes")

    blueSmarties = maskimageFrombinary(img, H_bin_dilate, 0)
    plt.figure()
    plt.imshow(blueSmarties, cmap="gray")
    plt.title("Smarties bleues")
    plt.show()
```

### Smarties jaunes

<p align="center">
  <image style="width:40%;" src="img/smarties_bin_green.png"></image> <image style="width:40%;" src="img/smarties_bin_erod_green.png"></image><image style="width:40%;" src="img/smarties_jaunes.png"></image>
  <div align="center" > 
    <i>Image binarisée précédente, image binarisée érodée et calque appliqué à l'image originale</i>
  </div>
</p>

On a ici appliqué une érosion, étant donné que le calque prenant les valeurs hautes, grâce à la commande `cv.erode`.

### Smarties bleues

<p align="center">
  <image style="width:40%;" src="img/smarties_bin_hue.png"></image> <image style="width:40%;" src="img/smarties_bin_dil_hue.png"></image><image style="width:40%;" src="img/smarties_bleues.png"></image>
  <div align="center" > 
    <i>Image binarisée précédente, image binarisée érodée et calque appliqué à l'image originale</i>
  </div>
</p>

On a ici appliqué une dilatation, étant donné que le calque prenant les valeurs basses, grâce à la commande `cv.dilate`.

## Elimination du ciel de l'image CeriserP

Le canal le plus judicieux serait d'utiliser le même que pour les smarties bleues (le ciel étant bleu lui aussi), à savoir le canal H. Voici nos résultats :

```python
from scipy import signal
import numpy as np

from matplotlib import pyplot as plt
import cv2 as cv

def histogramLvlOfGrey(img):
    hist = np.zeros(256)
    for line in img:
        for pixel in line:
            hist[pixel] += 1
    return hist

def statisticImageMoments(img,hist,i):
    sum = 0
    for j in range(0,256):
        sum += hist[j]*(j)**i
    return 1/(img.shape[0]*img.shape[1]) * sum

def binaryImg(img, seuil):

    imgB = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > seuil:
                imgB[i,j] = 255
    return imgB
def autoBinaryImg(img):
    hist = histogramLvlOfGrey(img)
    sim = (statisticImageMoments(img, hist, 1))
    return binaryImg(img, sim)

def histogramRepartition(img):
    hist = histogramLvlOfGrey(img)
    Ymax = img.shape[0]
    Xmax = img.shape[1]
    N = Ymax*Xmax
    repartition = np.zeros_like(hist)
    kmax = np.where(hist == max(hist))[0][0]

    for i in range(1, kmax):
        repartition[i] = repartition[i-1] + hist[i]
    for i in range(0, kmax):
        repartition[i] = (kmax-1)*repartition[i]/N
    plt.bar(np.arange(256), repartition)
    newimg = np.zeros_like(img)
    for i in range(Ymax):
        for j in range(Xmax):
            img[i,j] = repartition[img[i,j]]
            gris = img[i,j]
            newGris = repartition[gris]
            newimg[i,j] = newGris
    return newimg

def maskimageFrombinary(img,mask,value):
    newimg = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i,j] == value:
                newimg[i,j] = img[i,j]
    return newimg

if __name__ == "__main__":
    plt.close("all")
    filename = "CerisierP.jpg"
    name = filename.split(".")[0]
    img = cv.imread("imagesTP/"+filename)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # Convert to RGB
    r,g,b = cv.split(img)
    h,s,v = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title("Original")

    plt.imshow(h, cmap="gray")
    plt.title("Hue")

    H_hist = histogramLvlOfGrey(h)
    plt.figure()
    plt.bar(np.arange(256), H_hist)
    plt.title("Histogramme du canal H")

    H_bin = binaryImg(h, 25)
    plt.figure()
    plt.imshow(H_bin, cmap="gray")
    plt.title("Binarisation du canal H")

    cerisier = maskimageFrombinary(img, H_bin, 255)
    plt.figure()
    plt.imshow(cerisier)
    plt.title("Cerisier sans fond")

    img2 = cv.imread("imagesTP/"+"DJI_0093.JPG")
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB) # Convert to RGB

    img2crop = img2[0:img.shape[0],0:img.shape[1]]
    img2mask = maskimageFrombinary(img2crop, H_bin, 0)

    plt.figure()
    plt.imshow(img2mask + cerisier)
    plt.title("Cerisier sur fond")

    plt.show()
```

<p align="center">
  <image style="width:40%;" src="img/cerisier.png"></image> <image style="width:40%;" src="img/cerisier_hue.png"></image> <image style="width:40%;" src="img/cerisier_histo.png"></image> <image style="width:40%;" src="img/cerisier_sansf.png"></image> <image style="width:40%;" src="img/cerisier_fond.png"></image>
  <div align="center" > 
    <i>Image originale, image du canal H, histogramme du canal H, image originale sans fond et image originale avec fond de l'image "DJI_0093.jpg"</i>
  </div>
</p>

Comme pour les segmentations précédentes, regarder l'histogramme nous donne une idée de la valeur du seuil (ici à 25), et après quelques calques, le fond de l'image peut être enlevé et remplacé par une autre couleur ou image.


## Segmentation d'autres images

Voici le code utilisé :

```python
from scipy import signal
import numpy as np

from matplotlib import pyplot as plt
import cv2 as cv

def histogramLvlOfGrey(img):
    hist = np.zeros(256)
    for line in img:
        for pixel in line:
            hist[pixel] += 1
    return hist

def statisticImageMoments(img,hist,i):
    sum = 0
    for j in range(0,256):
        sum += hist[j]*(j)**i
    return 1/(img.shape[0]*img.shape[1]) * sum

def binaryImg(img, seuil):

    imgB = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > seuil:
                imgB[i,j] = 255
    return imgB
def autoBinaryImg(img):
    hist = histogramLvlOfGrey(img)
    sim = (statisticImageMoments(img, hist, 1))
    return binaryImg(img, sim)

def histogramRepartition(img):
    hist = histogramLvlOfGrey(img)
    Ymax = img.shape[0]
    Xmax = img.shape[1]
    N = Ymax*Xmax
    repartition = np.zeros_like(hist)
    kmax = np.where(hist == max(hist))[0][0]

    for i in range(1, kmax):
        repartition[i] = repartition[i-1] + hist[i]
    for i in range(0, kmax):
        repartition[i] = (kmax-1)*repartition[i]/N
    plt.bar(np.arange(256), repartition)
    newimg = np.zeros_like(img)
    for i in range(Ymax):
        for j in range(Xmax):
            img[i,j] = repartition[img[i,j]]
            gris = img[i,j]
            newGris = repartition[gris]
            newimg[i,j] = newGris
    return newimg

def maskimageFrombinary(img,mask,value):
    newimg = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i,j] == value:
                newimg[i,j] = img[i,j]

    return newimg

if __name__ == "__main__":
    plt.close("all")
    filename = "CerisierP.jpg"
    name = filename.split(".")[0]
    img = cv.imread("imagesTP/"+filename)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # Convert to RGB
    r,g,b = cv.split(img)
    h,s,v = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))


    H_bin = binaryImg(h, 25)
    cerisier = maskimageFrombinary(img, H_bin, 255)

    img2 = cv.imread("imagesTP/"+"DJI_0093.JPG")
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB) # Convert to RGB

    img2crop = img2[0:img.shape[0],0:img.shape[1]]
    img2mask = maskimageFrombinary(img2crop, H_bin, 0)

    img3 = cv.imread("imagesTP/"+"coquelicots.jpg")
    img3 = cv.cvtColor(img3,cv.COLOR_BGR2RGB) # Convert to RGB
    r,g,b = cv.split(img3)
    h,s,v = cv.split(cv.cvtColor(img3, cv.COLOR_RGB2HSV))
    plt.figure()
    plt.imshow(img3)
    plt.title("Coquelicots")
    
    G_hist = histogramLvlOfGrey(g)
    plt.figure()
    plt.bar(np.arange(256), G_hist)
    plt.title("Histogramme de l'image")

    G_bin = binaryImg(g, 30)
    G_bin = cv.dilate(G_bin, np.ones((3,3),np.uint8), iterations=1)
    
    plt.figure()
    plt.imshow(g, cmap="gray")
    plt.title("Canal G")
    
    plt.figure()
    plt.imshow(G_bin, cmap="gray")
    plt.title("Image binairisée")

    coquelicots = maskimageFrombinary(img3, G_bin, 0)
    plt.figure()
    plt.imshow(coquelicots)
    plt.title("Coquelicots sans fond")

    greenImg = np.zeros_like(coquelicots)
    greenImg[:, :, 1] = 100

    plt.figure()
    plt.imshow(greenImg)
    plt.title("Image verte")


    greenImg = maskimageFrombinary(greenImg, G_bin, 255)
    coquelicotsOnGreen = greenImg + coquelicots
    plt.figure()
    plt.imshow(coquelicotsOnGreen)
    plt.title("Coquelicots sur fond vert")

    plt.show()
```

### Segmentation

Comme pour les segmentations précédentes, on a utilisé les histogrammes pour vérifier le canal :

<p align="center">
  <image style="width:40%;" src="img/clquelicots.png"></image> <image style="width:40%;" src="img/clquelicots-G.png"></image> <image style="width:40%;" src="img/clquelicots_hist.png"></image> <image style="width:40%;" src="img/clquelicots_bin.png"></image> 
  <div align="center" > 
    <i>Image originale, image du canal H, histogramme du canal H, image originale sans fond et image originale avec fond de l'image "DJI_0093.jpg"</i>
  </div>
</p>




### Modification de la couleur sur l'image couleur

Il suffisait ensuite d'appliquer le calque à l'image originale et d'ensuite ajouter une couleur générée avec les commandes `greenImg = np.zeros_like(coquelicots)` et `greenImg[:, :, 1] = 100`. On obtiens alors les images suivantes :

<p align="center">
  <image style="width:40%;" src="img/clquelicots_sansf.png"></image> <image style="width:40%;" src="img/clquelicots_fvert.png"></image>
  <div align="center" > 
    <i>Image originale, image du canal H, histogramme du canal H, image originale sans fond et image originale avec fond de l'image "DJI_0093.jpg"</i>
  </div>
</p>

