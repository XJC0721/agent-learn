import { app, BrowserWindow, shell, ipcMain, Notification } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { readFileSync, writeFileSync, existsSync } from 'fs'

let dataFilePath

function getDataPath() {
  return join(app.getPath('userData'), 'pomodoro-records.json')
}

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 420,
    height: 700,
    minWidth: 380,
    minHeight: 650,
    show: false,
    frame: false,
    resizable: true,
    autoHideMenuBar: true,
    backgroundColor: '#1a1a2e',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

function registerIpcHandlers() {
  dataFilePath = getDataPath()

  ipcMain.handle('storage:load-records', () => {
    if (!existsSync(dataFilePath)) return []
    try {
      return JSON.parse(readFileSync(dataFilePath, 'utf-8'))
    } catch {
      return []
    }
  })

  ipcMain.handle('storage:save-record', (_, record) => {
    let records = []
    if (existsSync(dataFilePath)) {
      try { records = JSON.parse(readFileSync(dataFilePath, 'utf-8')) } catch {}
    }
    records.push({ ...record, savedAt: new Date().toISOString() })
    writeFileSync(dataFilePath, JSON.stringify(records, null, 2), 'utf-8')
    return true
  })

  ipcMain.handle('storage:clear-records', () => {
    writeFileSync(dataFilePath, '[]', 'utf-8')
    return true
  })

  ipcMain.handle('notification:send', (_, { title, body }) => {
    if (Notification.isSupported()) {
      new Notification({ title, body }).show()
    }
    return true
  })

  ipcMain.on('app:minimize', (event) => {
    BrowserWindow.fromWebContents(event.sender)?.minimize()
  })

  ipcMain.on('app:close', (event) => {
    BrowserWindow.fromWebContents(event.sender)?.close()
  })
}

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.pomodoro.desktop')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  registerIpcHandlers()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
