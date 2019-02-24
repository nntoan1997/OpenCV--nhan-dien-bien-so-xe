from tkinter import *

class Root(Tk):
    def _init_(self):
        super(Root, self)._init_()
        self.title("tkinter")
        self.minsize(640,400)
        self.vm_iconbitmap('icon.ico')

if _name_ == '_main_':
    root = Root()
    root.mainloop()
