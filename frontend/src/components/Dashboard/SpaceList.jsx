import { motion } from 'framer-motion';
import PropTypes from 'prop-types';

const SpaceList = ({ spaces }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'available': return 'bg-green-500';
      case 'occupied': return 'bg-red-500';
      case 'uncertain': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'available': return 'Available';
      case 'occupied': return 'Occupied';
      case 'uncertain': return 'Uncertain';
      default: return 'Unknown';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h3 className="text-xl font-semibold mb-4">Parking Spaces</h3>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {spaces.map((space, index) => (
          <motion.div
            key={space.space_id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg hover:bg-gray-900 transition-colors"
          >
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${getStatusColor(space.status)}`} />
              <div>
                <p className="font-medium text-white">{space.space_id}</p>
                <p className="text-sm text-gray-400">
                  {space.dimensions?.length?.toFixed(1)}m × {space.dimensions?.width?.toFixed(1)}m
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-white">{getStatusText(space.status)}</p>
              <p className="text-xs text-gray-400">{(space.confidence * 100).toFixed(0)}% confidence</p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

SpaceList.propTypes = {
  spaces: PropTypes.arrayOf(
    PropTypes.shape({
      space_id: PropTypes.string.isRequired,
      status: PropTypes.string.isRequired,
      confidence: PropTypes.number.isRequired,
      dimensions: PropTypes.shape({
        length: PropTypes.number,
        width: PropTypes.number,
      }),
    })
  ).isRequired,
};

export default SpaceList;
