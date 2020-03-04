# franks-ml-model

Example ML model as described in the blog posts:

- [How to Package your ML Project for Easy Distribution](https://ml-illustrated.github.io/2020/02/15/packaging-your-ml-project-for-faster-deployment.html) 
- [How to (Easily) Add Tests to your ML Projects](https://ml-illustrated.github.io/2020/02/24/adding-tests-to-your-ml-package.html)

## Features

```python
from franks_ml_model import EfficientNetInfer
model = EfficientNetInfer()
fn_image='tests/test_files/dog.jpg'
top_predictions = model.infer_image( fn_image )
# [[207, 'golden retriever', 0.5610832571983337], [213, 'Irish setter, red setter', 0.22866328060626984],...
```


## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
