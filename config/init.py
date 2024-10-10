import os

def set_pythonpath():
    """Ajoute la racine du projet au PYTHONPATH."""
    # Calcul du chemin vers la racine du projet à partir de `config`
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    venv_path = os.getenv('VIRTUAL_ENV')
    if not venv_path:
        print("⚠️  Aucun environnement virtuel activé.")
        return

    activate_script = os.path.join(venv_path, 'bin', 'activate')

    # Ajout du PYTHONPATH dans le script d'activation de l'environnement virtuel
    with open(activate_script, 'a') as f:
        f.write(f'\n# Ajout automatique du PYTHONPATH pour le projet\n')
        f.write(f'export PYTHONPATH="{project_root}:$PYTHONPATH"\n')

    print(f"✅ PYTHONPATH ajouté à l'environnement virtuel : {project_root}")

def main():
    set_pythonpath()

if __name__ == "__main__":
    main()
