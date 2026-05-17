import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

const api = {
  saveRecord: (record) => ipcRenderer.invoke('storage:save-record', record),
  loadRecords: () => ipcRenderer.invoke('storage:load-records'),
  clearRecords: () => ipcRenderer.invoke('storage:clear-records'),
  sendNotification: (title, body) => ipcRenderer.invoke('notification:send', { title, body }),
  minimizeWindow: () => ipcRenderer.send('app:minimize'),
  closeWindow: () => ipcRenderer.send('app:close')
}

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electronAPI', api)
    contextBridge.exposeInMainWorld('electron', electronAPI)
  } catch (e) {
    console.error(e)
  }
} else {
  window.electronAPI = api
  window.electron = electronAPI
}
