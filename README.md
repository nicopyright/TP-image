<link href="style.css" rel="stylesheet"></link>

# Prise en main "rapide"
<div align="justify">
- Tester le programme qui lit puis affiche une image couleur ainsi que ses trois composantes. Les tests seront réalisés à partir des images disponibles dans le répertoire fourni.

<p align="center" >
    <image style="width:80%;" src="img/CerisierP_couleurs.png" ></image>
  <figcaption align="center" class="caption"> 
    Image d'origine et ses trois composantes du fichier "CerisierP.jpg"
  </figcaption>
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
  <figcaption align="center" class="caption"> 
    Histogramme de l'image "cerisierp.jpg"
  </figcaption>
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
  <figcaption align="center" class="caption"> 
    Image binarisée avec seuil calculé de l'image "CerisierP.jpg"
  </figcaption>
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
  <figcaption align="center" class="caption"> 
    Image et spectre 2D de l'image générée avec les paramètres f1=0.1 et f2=0
  </figcaption>
</p>


<p align="center">
  <image style="width:42%;" src="img/image2.png"></image> <image style="width:40%;" src="img/spectre2.png"></image>
  <figcaption align="center" class="caption"> 
  Image et spectre 2D de l'image générée avec les paramètres f1=0 et f2=0.1
  </figcaption>
</p>

<p align="center">
  <image style="width:42%;" src="img/image3.png"></image> <image style="width:40%;" src="img/spectre3.png"></image>
  <figcaption align="center" class="caption"> 
  Image et spectre 2D de l'image générée avec les paramètres f1=0.3 et f2=0.3
  </figcaption>
</p>

<p align="center">
<image style="width:42%;" src="img/image4.png"></image> <image style="width:40%;" src="img/spectre4.png"></image>
  <figcaption align="center" class="caption"> 
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
  <figcaption align="center" class="caption"> 
    Image et spectre 2D de l'image "contour vertical"
  </figcaption>
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
  <figcaption align="center" class="caption"> 
    Image et spectre 2D de l'image "contour horizontal"
  </figcaption>
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
  <figcaption align="center" class="caption"> 
    Image et spectre 2D de l'image "contour oblique"
  </figcaption>
</p>
On observe un contour oblique dans l'image, ce qui se traduit par une ligne horizontale, une ligne verticale et une ligne oblique opposée à celle de l'image dans son spectre. Théoriquement, il n'y aurais qu'une ligne oblique mais étant donné que l'image est discrète et composée de pixels, la ligne "oblique" visible est en fait un escalier composé de lignes horizontales et verticales.

### c. Texture

Toutes les images de cette partie ont étés mises en nuances de gris pour des raisons de clareté et de code.

#### Metal0007GP
<p align="center">
  <image style="width:42%;" src="img/imageMetal.png"></image> <image style="width:40%;" src="img/imageMetal_spectre2D.png"></image>
  <figcaption align="center" class="caption"> 
    Image et spectre 2D de l'image "Metal0007GP"
  </figcaption>
</p>

On peut observer des lignes horizontales et verticales dans l'image, ce qui se traduit dans le spectre par des lignes verticales et horizontales. On peut même distinguer l'inclinaison des lignes de l'image depuis les lignes du spectre. 

#### Water0000GP
<p align="center">
  <image style="width:42%;" src="img/imageWater.png"></image> <image style="width:40%;" src="img/imageWater_spectre2D.png"></image> 
  <figcaption align="center" class="caption"> 
    Image et spectre 2D de l'image "Water0000GP"
  </figcaption>
</p>

Contrairement à l'image précédente, cette image est plus diffuse. On peut néanmoins remarquer que les vagues sont des formes horizontales. Ces formes se traduisent dans le spectre par une forme de sablier vertical

#### Leaves0012GP
<p align="center">
  <image style="width:42%;" src="img/imageLeaves.png"></image> <image style="width:40%;" src="img/imageLeaves_spectre2D.png"></image>
  <figcaption align="center" class="caption"> 
    Image et spectre 2D de l'image "Leaves0012GP"
  </figcaption>
</p>

\\\Ajouter description

## 2. Phénomène de repliement

#### Avec image en 128x128 et Fe=1

<p align="center">
  <image style="width:43%;" src="img/imageRepliement128.png"></image> <image style="width:40%;" src="img/imageRepliement128_spectre2D.png"></image>
  <figcaption align="center" class="caption"> 
    Image et spectre 2D de l'image générée avec <code>atom(128,128,0.15,0.37)</code>
  </figcaption>
</p>

#### Avec image en 64x64 et Fe=0.5

<p align="center">
  <image style="width:43%;" src="img/imageRepliement64.png"></image> <image style="width:40%;" src="img/imageRepliement64_spectre2D.png"></image>
  <figcaption align="center" class="caption"> 
    Image et spectre 2D de l'image générée avec <code >atom(64,64,0.15,0.37)</code >
  </figcaption>
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
  <figcaption align="center" class="caption"> 
    Images et composantes pour arriver à l'image IHSL
  </figcaption>
</p>

Quelle sont les limites de l’espace HSV ?



# IV. Segmentation d’images
