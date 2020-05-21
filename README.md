# avito-category

Автоматический определитель категории для объявления из авито.

---

## Запуск

Для запуска скачайте два файла:

- [word2vec](https://drive.google.com/file/d/1IMu8z-c9X880irAl0TMBh9DCMr61ISM0/view?usp=sharing)
- [udpipe препроцесс](https://drive.google.com/file/d/1jVERjmH9auxcuvLO7xQw9590Pl-jYWjN/view?usp=sharing)

Поместите их в папку `avitoServer/model_files` рядом с другими файлами.

Теперь можно запускать сервер в тестовом режиме:

```bash
$ cd avitoServer
$ python manage.py runserver
```