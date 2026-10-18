"""Analyse automatique d'une mosaique : fissures, colmatage, mesures.

    masque.py      contour de l'echantillon ; les bords ne sont pas des fissures
    donnees.py     annotations (lecture seule), decoupage apprentissage / test
    reseau.py      nettoyage des traces detectes (bords, raccords entre tuiles)
    bandes.py      bandes redressees le long des fissures, a pleine resolution
    classement.py  etat du colmatage tous les 10 um, lisse le long de la fissure
    evaluation.py  mesures sur les tuiles de test, jamais apprises
    chaine.py      la mosaique entiere, de l'image aux longueurs

Les annotations humaines ne sont jamais modifiees : les resultats vont dans
`annotation_app/data/analyse/<image>/`.
"""
