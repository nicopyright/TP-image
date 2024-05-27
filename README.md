<link href="style.css" rel="stylesheet"></link>

# Prise en main "rapide"

- Tester le programme qui lit puis affiche une image couleur ainsi que ses trois composantes. Les tests seront réalisés à partir des images disponibles dans le répertoire fourni.

  \\\ Utiliser le code de bas pour les images de couleurs
  
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

\\\ Utiliser le code ci dessus pour avoir l'image de l'histogramme
  
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
<p align="center">
  <image style="width:50%;" src="https://github.com/nicopyright/TP-image/assets/104890990/fb4b078e-87e0-4965-a01d-22e135202d1f"></image>
</p>



- Écrire un programme qui réalise les opérations suivantes :
  
-- Calcul et affichage de l’histogramme d’une image (appel de la fonction   histogramme réalisée en question 3)
-- Égalisation d’histogramme sur cette image
-- Affichage de la fonction de répartition, de l’histogramme de l’image égalisée.

# II. Transformée de Fourier

### a. Harmoniques pures


<p align="center" style="display:inline-block;">
    <image style="width:42%;" src="img/image1.png" ></image> <image style="width:40%;" src="img/spectre1.png"></image>
  <figcaption align="center">
    Image et spectre 2D de l'image générée avec les paramètres f1=0.1 et f2=0
  </figcaption>
</p>


<p align="center">
  <image style="width:42%;" src="img/image2.png"></image> <image style="width:40%;" src="img/spectre2.png"></image>
  <figcaption align="center">
  Image et spectre 2D de l'image générée avec les paramètres f1=0 et f2=0.1
  </figcaption>
</p>

<p align="center">
  <image style="width:42%;" src="img/image3.png"></image> <image style="width:40%;" src="img/spectre3.png"></image>
  <figcaption align="center">
  Image et spectre 2D de l'image générée avec les paramètres f1=0.3 et f2=0.3
  </figcaption>
</p>

<p align="center">
<image style="width:42%;" src="img/image4.png"></image> <image style="width:40%;" src="img/spectre4.png"></image>
  <figcaption align="center">
  Image et spectre 2D de l'image générée avec les paramètres f1=-0.3 et f2=0.1
  </figcaption>
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
  <figcaption align="center">
    Image et spectre 2D de l'image "contour vertical"
  </figcaption>
</p>

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
  <figcaption align="center">
    Image et spectre 2D de l'image "contour horizontal"
  </figcaption>
</p>

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
  <figcaption align="center">
    Image et spectre 2D de l'image "contour oblique"
  </figcaption>
</p>

### c. Texture

#### Metal0007G
<p align="center">
  <image style="width:42%;" src="img/imageMetal.png"></image> <image style="width:40%;" src="img/imageMetal_spectre2D.png"></image>
  <figcaption align="center">
    Image, spectre 2D et spectre 3D de l'image "Metal0007GP"
  </figcaption>
</p>

\\\Ajouter description

#### Water0000G
<p align="center">
  <image style="width:42%;" src="img/imageWater.png"></image> <image style="width:40%;" src="img/imageWater_spectre2D.png"></image> 
  <figcaption align="center">
    Image, spectre 2D et spectre 3D de l'image "Water0000GP"
  </figcaption>
</p>

\\\Ajouter description

#### Leaves0012G
<p align="center">
  <image style="width:42%;" src="img/imageLeaves.png"></image> <image style="width:40%;" src="img/imageLeaves_spectre2D.png"></image>
  <figcaption align="center">
    Image, spectre 2D et spectre 3D de l'image "Leaves0012GP"
  </figcaption>
</p>

\\\Ajouter description

## 2. Phénomène de repliement

# III. Changement d’espaces colorimétriques (comparaison HSV/IHLS)

# IV. Segmentation d’images
