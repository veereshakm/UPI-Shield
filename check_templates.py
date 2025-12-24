import os
from jinja2 import Environment, FileSystemLoader

def check_templates():
    template_dir = 'templates'
    env = Environment(loader=FileSystemLoader(template_dir))
    
    files = [f for f in os.listdir(template_dir) if f.endswith('.html')]
    
    for f in files:
        try:
            env.parse(env.loader.get_source(env, f)[0])
            print(f"OK: {f}")
        except Exception as e:
            print(f"ERROR in {f}: {e}")

if __name__ == "__main__":
    check_templates()
