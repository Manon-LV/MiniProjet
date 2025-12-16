# MiniProjet

## Journalisation (Logging)

Tous les scripts de ce projet utilisent désormais le module `logging` de Python pour afficher les messages d'information, de debug et d'avertissement (warning). Aucun affichage direct avec `print` n'est utilisé pour les messages utilisateur ou de suivi d'exécution.

Pour voir les logs, configurez le niveau de logging souhaité en début de script, par exemple :

```python
import logging
logging.basicConfig(level=logging.INFO)  # ou DEBUG, WARNING
```

Les logs importants (info, warning, debug) apparaîtront dans la console selon le niveau choisi.