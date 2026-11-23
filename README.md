# SpeechaRT

Cette application Python de Praaline propose un flux volontairement réduit :

1. parcourir un corpus audio et ses annotations associées ;
2. visualiser la forme d’onde, le spectrogramme et les annotations d’un enregistrement ;
3. lancer Montreal Forced Aligner, Silence Detector ou CV and Syllable Segmenter
   depuis l’onglet **Plugins**.

Elle ne modifie pas l’application C++/Qt existante. MFA et ses modèles restent des
prérequis externes ; les données de travail MFA sont isolées dans un dossier
temporaire propre à chaque alignement. La visualisation utilise NumPy et
SoundFile ; la lecture audio utilise SoundDevice. Installez les dépendances avec
`python3 -m pip install -r requirements.txt`.

## Lancer l’application

Utilisez un Python avec Tkinter et indiquez l’exécutable `mfa` de l’environnement
Conda où MFA est installé :

```sh
python3 -m SpeechaRT
```

Depuis le dossier `SpeechaRT`, le module est disponible directement. Sous
Windows, utilisez le Python de l’environnement MFA ; sous macOS ou Linux, activez
cet environnement avant de lancer la commande, ou sélectionnez le chemin complet
de l’exécutable MFA dans l’application.

Pour l’alignement MFA, la transcription peut être un fichier texte (`.txt` ou
`.lab`) ou un fichier Transcriber (`.trs`) ; dans ce dernier cas, le texte des
tours est transmis à MFA. Le premier onglet **Corpus** permet de
charger un dossier (recherche récursive des fichiers audio) ou d’ajouter des
audios individuellement, puis de retirer des entrées. Les fichiers TextGrid et
TRS ayant le même nom de base et se trouvant à côté d’un audio sont associés
automatiquement et affichés dans deux colonnes distinctes. Si les deux existent,
**Ouvrir la sélection** charge le TextGrid. Sélectionnez une entrée et choisissez
**Ouvrir la sélection**
(ou double-cliquez) pour passer à l’onglet **Visualisation** avec son audio et
ses annotations. La commande **Correction…** permet d’indiquer si la correction
est terminée et par qui ; ces informations sont conservées dans le fichier
`.praaline-corpus.json` du dossier du corpus. Les fichiers Transcriber `.trs` peuvent aussi être ouverts
séparément depuis le menu **Fichier** ; leurs tours et points de synchronisation
sont affichés sous forme d’intervalles dans un tier par locuteur.

Dans **Visualisation**, le bouton **Vidéo** affiche un panneau latéral. Choisissez
**Ouvrir une vidéo…** pour l’associer manuellement à l’audio courant ; cette
association est mémorisée pour les prochaines ouvertures. La vidéo suit la
position et la lecture de l’audio, et peut aussi être lue ou déplacée séparément
lorsque l’audio est arrêté. Les formats courants MP4, MOV, M4V, AVI, MKV et WebM
sont proposés ; leur lecture dépend des codecs reconnus par OpenCV.

À la première ouverture d’un dossier, Praaline parcourt les audios et crée
`praaline-corpus.csv` à sa racine. Les ouvertures suivantes réutilisent ce
catalogue pour éviter le parcours récursif ; **Actualiser** rescane le dossier
pour repérer les nouveaux fichiers et met à jour le CSV. Le catalogue contient
des chemins relatifs, les annotations associées et les états de correction :
vous pouvez le transmettre avec le corpus, en conservant la même structure de
dossiers. Les corrections restent aussi compatibles avec les anciens fichiers
`.praaline-corpus.json`.

Chaque enregistrement de **Correction…** archive l’état courant des fichiers TRS
et TextGrid associés dans `.praaline-history`, à l’intérieur du dossier du
corpus. Les états identiques ne sont pas dupliqués. **Historique…** affiche les
versions et permet de restaurer une ancienne ; l’état courant est conservé dans
l’historique avant la restauration. Les annotations sont archivées telles
qu’elles se trouvent sur le disque au moment de l’enregistrement de la
correction ; les modifications faites dans un autre outil doivent donc être
suivies d’un enregistrement de **Correction…** pour créer leur version.

L’écran de visualisation est consacré au signal acoustique. Les menus
**Fichier > Ouvrir un fichier audio…** et **Fichier > Ouvrir un TextGrid…**
permettent aussi de charger ces ressources séparément. Utilisez **Fermer les
annotations** pour masquer les tiers. Dans la visualisation, cochez ou décochez
la case à gauche du nom de chaque niveau pour afficher ou masquer ses
annotations. Les niveaux masqués se réduisent et les niveaux visibles
s’agrandissent pour faciliter la lecture ; le nom et la case de chaque niveau
restent accessibles. Chaque niveau visible a sa propre couleur, reprise sur son
nom, son fond léger et ses frontières d’intervalles.

L’onglet **Plugins** présente les enregistrements du corpus, les plugins
disponibles et les paramètres du plugin sélectionné. Choisissez un audio puis
**Montreal Forced Aligner (MFA)** ; si un fichier `.txt` ou `.lab` du même nom
existe à côté de l’audio, il est proposé en priorité ; sinon le fichier `.trs`
associé est proposé. Les fichiers `.trs` peuvent aussi être sélectionnés dans
le champ de transcription. Ses paramètres restent modifiables dans cet onglet.
En cas d’échec, un journal `*.mfa-error-*.log` est enregistré dans le dossier de
sortie. **Silence Detector** détecte les pauses
par fenêtres RMS de 10 ms, avec un seuil adaptatif au bruit de fond (limite par
défaut : -35 dB) et une durée minimale par défaut de 0,3 seconde. Il crée un
fichier `.trs` à côté de l’audio, avec les pauses marquées `#` et un locuteur
générique « Locuteur 1 ». Si le fichier TRS existe déjà, l’application demande
confirmation avant de le remplacer et archive les annotations précédentes dans
`.praaline-history`. Les résultats sont associés à l’enregistrement et chargés
dans la visualisation s’il est toujours affiché. La commande
**CV and Syllable Segmenter** nécessite un TextGrid associé à l’audio contenant
des niveaux d’intervalles `phone` et `token` ; les niveaux MFA `phones` et
`words` sont également reconnus. Il crée ou remplace les niveaux
`CV` (suites de consonnes/voyelles) et `syll` (syllabes déterminées par les
noyaux vocaliques et l’attaque maximale). Les niveaux d’entrée et de sortie,
l’inventaire des voyelles IPA et le nombre maximal de consonnes en attaque
(1 à 3) sont configurables. Avant l’écriture, le TextGrid existant est archivé
dans `.praaline-history` ; les résultats sont chargés dans la visualisation si
l’enregistrement est toujours affiché. La commande **Plugins > Ouvrir les plugins**
conduit à cet onglet.
Les TextGrid Praat au format long affichent les tiers d’intervalles ; la
sélection temporelle et le zoom permettent d’examiner l’audio. Faites glisser
la souris sur le signal pour sélectionner une plage, puis appuyez sur **Entrée**
pour l’agrandir à la largeur de la vue. Le bouton **Ajuster** réaffiche ensuite
l’audio complet. Le bouton **Lire la vue** lit la sélection lorsqu’il y en a une,
sinon la portion temporelle actuellement visible ; il devient **Arrêter** pendant
la lecture. Le spectrogramme utilise un rendu phonétique en niveaux de gris,
avec la plage de fréquences limitée à 5 kHz et une fenêtre courte, comme le
spectrogramme Praaline. Le bouton **Options spectrogramme…** permet de choisir
la palette, l’échelle, la taille et le recouvrement de la fenêtre, l’affichage
des bins, l’échelle fréquentielle, la normalisation, le gain, le seuil et la
rotation des couleurs. Les cases **Normalised**, **Autoscaling** et **Niveau de
gris** contrôlent la normalisation par colonne, l’adaptation à la zone visible
et l’affichage monochrome ; les réglages sont appliqués directement, sans
bouton **Appliquer**. Le calcul du spectrogramme se fait en arrière-plan et la
forme d’onde est mise en cache pour accélérer les redessinages. Le rendu utilise
les réglages Praat de Praaline par défaut, dont l’**Autoscaling** de la zone
visible. La fréquence affichée commence au premier bin FFT, comme dans Praaline.
Les outils **Sélectionner**, **Effacer**, **Déplacer barre** et
**Annoter** permettent aussi de corriger les TextGrid et les fichiers TRS.
Sélectionnez une annotation pour mettre en évidence sa plage ; **Effacer** vide
son texte sans supprimer ses frontières temporelles. **Supprimer barre** retire
une frontière TextGrid et
fusionne les intervalles voisins en concaténant leur texte. **Ajouter barre**
scinde un intervalle TextGrid au clic, en laissant son texte dans l’intervalle
gauche. **Déplacer barre** permet de faire glisser une frontière TextGrid à
l’intérieur des intervalles voisins, ou une frontière TRS entre ses points Sync
voisins. Une limite de tour TRS est aussi déplaçable si le tour appartient à un
seul locuteur ; le texte reste conservé et aucun segment vide n’est créé.
Double-cliquez
sur une annotation pour modifier son texte sans changer ses frontières. Avec le
crayon, faites glisser sur un tier puis saisissez le texte. **Cmd+Z** sur macOS
(**Ctrl+Z** sous Windows/Linux) annule les dernières modifications d’annotation
dans le fichier ouvert, jusqu’à 20 actions. Sur un TextGrid, la
plage est découpée aux frontières
existantes et son texte est remplacé ; sur un TRS, un nouveau tour est ajouté
seulement dans une section existante et sans chevauchement avec un tour du même
locuteur. Chaque correction est enregistrée immédiatement. Pour une entrée de
corpus, l’état précédent est archivé avant la modification dans
`.praaline-history`. La touche **Espace**, quand le canevas de visualisation a le
focus, lance ou arrête la lecture de la vue. **Maj+←/→** passe à l’annotation
précédente ou suivante dans le même tier ; **Maj+↑/↓** passe au tier visible
précédent ou suivant et sélectionne l’annotation temporellement la plus proche.
La navigation ne boucle pas aux limites. L’application traite un audio à la fois.

## Tests

Les tests n’exécutent pas MFA et ne nécessitent pas de fichiers audio réels :

```sh
cd SpeechaRT
python3 -m unittest discover -s tests -v
```

Cette étape utilise NumPy et ne lance pas de lecture audio. NumPy, SoundFile et
SoundDevice sont requis pour l’application complète. La création d’un paquet
installable et la validation native de chaque système suivront séparément.
