import { create } from 'zustand';

export const useParkingStore = create((set) => ({
  spaces: [],
  stats: {
    total: 0,
    available: 0,
    occupied: 0
  },

  setSpaces: (spaces) => set({
    spaces,
    stats: {
      total: spaces.length,
      available: spaces.filter(s => s.status === 'available').length,
      occupied: spaces.filter(s => s.status === 'occupied').length
    }
  }),

  updateSpace: (spaceId, updates) => set((state) => ({
    spaces: state.spaces.map(space =>
      space.space_id === spaceId ? { ...space, ...updates } : space
    )
  })),

  applyChanges: (changes) => set((state) => {
    const updatedSpaces = [...state.spaces];
    changes.forEach(change => {
      const index = updatedSpaces.findIndex(s => s.space_id === change.space_id);
      if (index !== -1) {
        updatedSpaces[index] = {
          ...updatedSpaces[index],
          status: change.new_status,
          confidence: change.confidence
        };
      }
    });

    return {
      spaces: updatedSpaces,
      stats: {
        total: updatedSpaces.length,
        available: updatedSpaces.filter(s => s.status === 'available').length,
        occupied: updatedSpaces.filter(s => s.status === 'occupied').length
      }
    };
  })
}));
