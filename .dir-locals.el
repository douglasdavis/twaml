((python-mode . ((eval . (add-to-list 'eglot-server-programs '((python-mode)
                                                               "~/Software/Python/anaconda3/envs/twaml/bin/pyls")))
                 (eval . (setq py-shell-name  "~/Software/Python/anaconda3/envs/twaml/bin/python"
                               py-ipython-command "~/Software/Python/anaconda3/envs/twaml/bin/ipython"))
                 )))
