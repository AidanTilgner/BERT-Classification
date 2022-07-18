import { python } from 'pythonia'
// Import tkinter
const tk = await python('tkinter')
// All Python API access must be prefixed with await
const root = await tk.Tk()
// A function call with a $ suffix will treat the last argument as a kwarg dict
const a = await tk.Label$(root, { text: 'Hello World' })
await a.pack()
await root.mainloop()
python.exit() // Make sure to exit Python in the end to allow node to exit. You can also use process.exit.