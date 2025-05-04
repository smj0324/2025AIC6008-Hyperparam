import json
import subprocess

output = subprocess.check_output(['pip', 'list', '--format=json'], text=True)
packages = json.loads(output)

# pip로 설치된 것만 필터링
only_pip = [
    f"{pkg['name']}=={pkg['version']}"
    for pkg in packages
    if pkg.get('installer', 'pip') == 'pip'  # 대부분 installer 필드 없음 → pip 간주
]

# 파일로 저장
with open('requirements.txt', 'w') as f:
    f.write('\n'.join(only_pip))

print(f"{len(only_pip)} packages exported to requirements.txt")