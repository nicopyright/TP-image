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
```

Et pour nos tests, notre main : 

```python
if __name__ == "__main__":
    filename = "cerisierP.jpg"
    name = filename.split(".")[0]
    img = cv.imread("imagesTP/"+filename)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # Convert to RGB
    plt.imshow(img)
    hist = histogramLvlOfGrey(img)
    sim = (statisticImageMoments(img, hist, 1))
    print(sim)
    bI =binaryImg(img, sim)
    plt.imshow(bI, cmap="gray")
    plt.imsave(fname="imagesTP/output/" + name + "_binary.png", arr=bI, format="png", cmap="gray")
    plt.show()
```

Et voilà notre résultat sur l'image "" :
<p align="center">
  <image style="width:50%;" src="https://github.com/nicopyright/TP-image/assets/104890990/fb4b078e-87e0-4965-a01d-22e135202d1f"></image>
</p>



- Écrire un programme qui réalise les opérations suivantes :
  
-- Calcul et affichage de l’histogramme d’une image (appel de la fonction   histogramme réalisée en question 3)
-- Égalisation d’histogramme sur cette image
-- Affichage de la fonction de répartition, de l’histogramme de l’image égalisée.

# II. Transformée de Fourier

### a. Harmoniques pures

### b. Contour

### c. Texture

## 2. Phénomène de repliement

# III. Changement d’espaces colorimétriques (comparaison HSV/IHLS)

# IV. Segmentation d’images
