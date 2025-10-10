## Python 複数環境
### Mac
```bash
pyenv install 3.10.14
```

```bash
pyenv local 3.10.14
```

## venv
```bash
python -m venv .venv
```

#### Windows
```bash
.venv\Scripts\activate
```

#### Mac
```bash
source .venv/bin/activate
```

## Package Install
```bash
python -m pip install --upgrade pip
pip install --upgrade pip setuptools wheel
```

### Windows
```bash
python -m pip install -r requirements.txt
```

### Mac(Apple Silicon)
```bash
python -m pip install -r requirements_macos.txt
```

### 再インストール
### Windows
```bash
pip install --upgrade --force-reinstall --no-cache-dir -r requirements.txt
```

### Mac(Apple Silicon)
```bash
pip install --upgrade --force-reinstall --no-cache-dir -r requirements_macos.txt
```

### ワークスペース
- VS Code は新しいターミナルを開いたときに 自動で venv をアクティベートしないこと
- 「Python: Select Interpreter」を選ぶ

1. コマンドパレットを開く
2. 「Python: Select Interpreter」を選ぶ
3. .venv 内の Python を選択する

#### Windows
```bash
where python
```

#### Mac
```bash
which python
```

### ライブラリ確認
```bash
python -m pip list
```

## サーバ起動
```bash
uvicorn main:app --reload --port 8000  
```